#!/usr/bin/env python3
"""
audit_log.py — Structured audit logging (JSON Lines) for every evaluation.

Each evaluation appends exactly one JSON object to logs/inference_{YYYYMMDD}.jsonl.
This is the regulatory trail (AI Act / DORA): it lets an auditor reconstruct what
the system decided, over which images (by SHA256 hash), with which model version
and which triage rule — WITHOUT storing any PII (no plate, no name, no insured
description). Only internal identifiers (claim_id) and hashes are recorded.

Daily rotation is implicit: the filename's date is derived from the record's
timestamp, so a new file starts each day.

Thresholds/paths live in configs/audit_log.yaml (config out of code).

Public API
----------
    load_config(path=None) -> dict
    hash_bytes(data) -> str
    hash_image(path) -> str
    build_record(...) -> dict                 # pure, no I/O — PII-safe by design
    log_inference(...) -> Path                # build + append one JSONL line
    log_from_output(output, ...) -> Path      # extract from a validated output
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger("audit_log")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "audit_log.yaml"

# The only keys an audit record may carry. Anything else (and therefore any PII)
# is structurally excluded — build_record never copies arbitrary input through.
_ALLOWED_KEYS = {
    "timestamp",
    "id_evaluacion",
    "claim_id",
    "input_hashes",
    "model_version",
    "output_summary",
    "rule_id_applied",
    "processing_time_ms",
}


# ── Config ───────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> dict:
    """Load the audit-log config (log_dir, filename_pattern)."""
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Audit-log config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Hashing ──────────────────────────────────────────────────────────

def hash_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest of raw bytes (deterministic)."""
    return hashlib.sha256(data).hexdigest()


def hash_image(path) -> str:
    """Return the SHA256 hex digest of an image file's bytes.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found for hashing: {path}")
    return hash_bytes(path.read_bytes())


# ── Time / path helpers ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_token(timestamp: str) -> str:
    """Derive the YYYYMMDD token used for daily rotation from a timestamp."""
    date_part = timestamp[:10].replace("-", "")
    if len(date_part) == 8 and date_part.isdigit():
        return date_part
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _log_path(timestamp: str, log_dir: Optional[Path], config: dict) -> Path:
    base = Path(log_dir) if log_dir is not None else (PROJECT_ROOT / config.get("log_dir", "logs"))
    pattern = config.get("filename_pattern", "inference_{date}.jsonl")
    return base / pattern.format(date=_date_token(timestamp))


# ── Record building (pure) ───────────────────────────────────────────

def build_record(
    *,
    input_hashes: list,
    model_version: Union[dict, str],
    lane: str,
    rule_id: str,
    n_damages: int,
    processing_time_ms: int,
    total_eur: Optional[float] = None,
    id_evaluacion: Optional[str] = None,
    claim_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Build a PII-safe audit record. Pure: no I/O.

    Only whitelisted fields are emitted; there is no path for arbitrary metadata
    (and therefore PII) to leak into the log.
    """
    record = {
        "timestamp": timestamp or _now_iso(),
        "id_evaluacion": id_evaluacion,
        "claim_id": claim_id,
        "input_hashes": list(input_hashes),
        "model_version": model_version,
        "output_summary": {
            "lane": lane,
            "total_eur": total_eur,
            "n_damages": n_damages,
        },
        "rule_id_applied": rule_id,
        "processing_time_ms": processing_time_ms,
    }
    # Defensive invariant: never emit a key outside the whitelist.
    assert set(record) <= _ALLOWED_KEYS, f"audit record has non-whitelisted keys: {set(record) - _ALLOWED_KEYS}"
    return record


# ── Writing ──────────────────────────────────────────────────────────

def log_inference(
    *,
    input_hashes: list,
    model_version: Union[dict, str],
    lane: str,
    rule_id: str,
    n_damages: int,
    processing_time_ms: int,
    total_eur: Optional[float] = None,
    id_evaluacion: Optional[str] = None,
    claim_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    log_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Path:
    """Append one JSONL audit line for an evaluation and return the file path."""
    config = config if config is not None else load_config()
    record = build_record(
        input_hashes=input_hashes,
        model_version=model_version,
        lane=lane,
        rule_id=rule_id,
        n_damages=n_damages,
        processing_time_ms=processing_time_ms,
        total_eur=total_eur,
        id_evaluacion=id_evaluacion,
        claim_id=claim_id,
        timestamp=timestamp,
    )
    path = _log_path(record["timestamp"], log_dir, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    log.debug("audit line written to %s (%s)", path, record.get("id_evaluacion"))
    return path


def log_from_output(
    output: dict,
    log_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Path:
    """Append an audit line derived from a validated inference output (T1.2).

    Pulls input_hashes / processing_time_ms from output['audit'], the cost from
    output['estimacion'], and lane/rule from the top level.
    """
    audit = output.get("audit", {}) or {}
    estimacion = output.get("estimacion") or {}
    return log_inference(
        input_hashes=audit.get("input_hashes", []),
        model_version=output.get("model_version", {}),
        lane=output.get("lane", ""),
        rule_id=output.get("lane_rule_id", ""),
        n_damages=len(output.get("damages", []) or []),
        processing_time_ms=audit.get("processing_time_ms", 0),
        total_eur=estimacion.get("total_eur"),
        id_evaluacion=output.get("id_evaluacion"),
        claim_id=output.get("claim_id"),
        timestamp=output.get("timestamp"),
        log_dir=log_dir,
        config=config,
    )
