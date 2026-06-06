#!/usr/bin/env python3
"""
output_builder.py — Assemble and validate the canonical inference output (v1).

This is the single place where the pipeline's intermediate results (quality,
consolidated damages, alerts, cost estimate, triage decision, audit data) are
turned into the JSON contract that downstream insurance systems consume. Every
emitted object is validated against schemas/inference_output_v1.json BEFORE it
leaves the system; on failure we raise an explicit OutputValidationError rather
than emitting a malformed (and silently wrong) payload.

The schema is the source of truth (rule: explicit schemas for external output).
This module never relaxes it.

Public API
----------
    load_schema(path=None) -> dict
    generate_evaluation_id(timestamp=None) -> str
    utc_timestamp() -> str
    validate_output(output, schema=None) -> None        # raises on invalid
    build_output(**parts) -> dict                        # assemble + validate
"""

import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("output_builder")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = PROJECT_ROOT / "schemas" / "inference_output_v1.json"


class OutputValidationError(ValueError):
    """Raised when an assembled output does not conform to the JSON Schema.

    Subclasses ValueError so callers can catch it specifically or as a generic
    value error. The message lists every failing path so the failure is
    actionable (no silent drops).
    """


# ── Schema loading ───────────────────────────────────────────────────

def load_schema(path: Optional[Path] = None) -> dict:
    """Load the inference-output JSON Schema."""
    import json

    schema_path = Path(path) if path else DEFAULT_SCHEMA
    if not schema_path.exists():
        raise FileNotFoundError(f"Output schema not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _schema_version(schema: dict) -> str:
    """Read the pinned schema_version constant straight from the schema.

    Avoids hardcoding the version string in two places (the schema and here).
    """
    try:
        return schema["properties"]["schema_version"]["const"]
    except (KeyError, TypeError):
        return "1.0.0"


# ── Identifiers / time ───────────────────────────────────────────────

def utc_timestamp() -> str:
    """Return an RFC3339 UTC timestamp, e.g. '2026-06-06T20:00:00Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_evaluation_id(timestamp: Optional[str] = None) -> str:
    """Generate an evaluation id matching ^EVA-[0-9]{8}-[A-F0-9]{8}$.

    Format: EVA-YYYYMMDD-XXXXXXXX where the date comes from ``timestamp`` (or
    now, UTC) and the suffix is 8 random uppercase hex chars.
    """
    ts = timestamp or utc_timestamp()
    # Take the date part (first 10 chars 'YYYY-MM-DD'), strip the dashes.
    date_part = ts[:10].replace("-", "")
    if not (len(date_part) == 8 and date_part.isdigit()):
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(4).upper()  # 8 hex chars, A-F0-9
    return f"EVA-{date_part}-{suffix}"


# ── Validation ───────────────────────────────────────────────────────

def validate_output(output: dict, schema: Optional[dict] = None) -> None:
    """Validate ``output`` against the schema; raise OutputValidationError if invalid.

    Raises:
        OutputValidationError: with a message listing every failing field path.
    """
    from jsonschema import Draft202012Validator, FormatChecker

    schema = schema if schema is not None else load_schema()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path))
    if errors:
        lines = []
        for err in errors:
            location = "/".join(str(p) for p in err.absolute_path) or "<root>"
            lines.append(f"  - {location}: {err.message}")
        raise OutputValidationError(
            "Inference output failed schema validation "
            f"({len(errors)} error(s)):\n" + "\n".join(lines)
        )


# ── Assembly ─────────────────────────────────────────────────────────

def build_output(
    *,
    claim_id: str,
    model_version: dict,
    quality: dict,
    damages: list,
    estimacion: dict,
    lane: str,
    lane_rule_id: str,
    lane_reason: str,
    next_action: str,
    audit: dict,
    alerts: Optional[list] = None,
    zones_summary: Optional[dict] = None,
    id_evaluacion: Optional[str] = None,
    timestamp: Optional[str] = None,
    schema: Optional[dict] = None,
    validate: bool = True,
) -> dict:
    """Assemble the canonical inference output and validate it against the schema.

    The intermediate parts (quality, damages, estimacion, alerts, audit, ...) are
    produced by upstream pipeline stages and passed in already shaped per the
    schema. This function only composes the top-level object, fills in
    schema_version / id_evaluacion / timestamp, and validates.

    Args:
        claim_id: Claim id in the case-management system (no PII).
        model_version: {damage_model, parts_model, ...}.
        quality: {valid, per_image:[...]}.
        damages: Consolidated damages list (may be empty).
        estimacion: Cost estimate object.
        lane / lane_rule_id / lane_reason / next_action: Triage decision.
        audit: {input_hashes, processing_time_ms, ...}.
        alerts: Alerts list (defaults to []).
        zones_summary: Optional per-zone damage counts.
        id_evaluacion: Optional explicit id (generated if None).
        timestamp: Optional explicit RFC3339 UTC timestamp (generated if None).
        schema: Optional pre-loaded schema (loaded if None).
        validate: If True (default), validate before returning.

    Returns:
        The validated output dict.

    Raises:
        OutputValidationError: if the assembled output is invalid.
    """
    schema = schema if schema is not None else load_schema()
    ts = timestamp or utc_timestamp()
    eval_id = id_evaluacion or generate_evaluation_id(ts)

    output = {
        "schema_version": _schema_version(schema),
        "id_evaluacion": eval_id,
        "timestamp": ts,
        "model_version": model_version,
        "claim_id": claim_id,
        "quality": quality,
        "damages": damages,
        "alerts": alerts if alerts is not None else [],
        "estimacion": estimacion,
        "lane": lane,
        "lane_rule_id": lane_rule_id,
        "lane_reason": lane_reason,
        "next_action": next_action,
        "audit": audit,
    }
    # zones_summary is optional in the schema; only include it when provided.
    if zones_summary is not None:
        output["zones_summary"] = zones_summary

    if validate:
        validate_output(output, schema=schema)
        log.debug("Output %s validated OK against schema %s", eval_id, _schema_version(schema))

    return output
