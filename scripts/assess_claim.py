#!/usr/bin/env python3
"""
assess_claim.py — Single operational entry point for a parking-damage claim.

Wires the Sprint 1 stages into one deterministic pipeline:

    images → hash → quality gate → damage inference (valid images only)
           → aggregation (placeholder, T2.5) → cost estimate (placeholder, T2.2)
           → deterministic triage (T1.3) → schema-validated output (T1.2)
           → JSONL audit line (T1.4)

The model never decides anything operational here: it only supplies damage
candidates; the lane is decided by deterministic rules.

Model-dependent stages (damage detector, the quality gate's vehicle detector)
are INJECTABLE so the whole pipeline runs end-to-end offline in tests. The
production defaults lazily load the real models and fail with a clear message if
the weights are absent (e.g. baseline_v1.0/best.pt still on the remote GPU box).

Placeholders are explicit and marked TODO with the task that will replace them:
- aggregation does not yet deduplicate across views (TODO T2.5);
- damages carry placeholder zone/part/severity (TODO T2.3 + parts via localize);
- cost estimate is zeroed with confidence 0 (TODO T2.2) — so a claim can never
  be auto-resolved to green until real costing exists (conservative).

CLI
---
    python scripts/assess_claim.py --claim-id SIN-1 --images dir/ --metadata m.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

# Local modules (scripts/ is on sys.path when run as a script; tests add it too).
import audit_log
import output_builder
import quality_gate
import triage

log = logging.getLogger("assess_claim")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = "assess_claim/0.1.0"

_DAMAGE_TYPES = {"dent", "scratch", "crack", "broken_light"}


# ── Placeholder stages (replaced by later Sprint-2 tasks) ────────────

def _extension_from_area(area_pct: Optional[float]) -> str:
    """Heuristic placeholder extension from damage area %. TODO T2.3: use the
    severity matrix and the damaged part's area instead of a global %."""
    if area_pct is None:
        return "small"
    if area_pct < 2:
        return "small"
    if area_pct < 10:
        return "medium"
    return "large"


def _aggregate(per_image: list) -> list:
    """Flatten per-image damage candidates into consolidated damages.

    PLACEHOLDER (TODO T2.5): no cross-view deduplication yet — each candidate
    becomes its own consolidated damage. Schema-required fields not computed at
    this stage (zone/part/severity) are filled with safe placeholders.
    """
    damages = []
    idx = 1
    for image_hash, raw_list in per_image:
        for raw in raw_list:
            dtype = raw.get("type")
            if dtype not in _DAMAGE_TYPES:
                continue  # ignore anything outside the trained classes
            damages.append({
                "damage_id": f"D{idx}",
                "type": dtype,
                "zone": raw.get("zone", "unknown"),                 # TODO T1.5+: via localize.py
                "part": raw.get("part", "unknown"),
                "extension": raw.get("extension") or _extension_from_area(raw.get("area_pct")),
                "severity": raw.get("severity", "leve"),            # TODO T2.3: severity matrix
                "confidence": float(raw.get("confidence", 0.0)),
                "supporting_images": [image_hash],
                "structural_suspicion": bool(raw.get("structural_suspicion", False)),
            })
            idx += 1
    return damages


def _placeholder_estimacion(damages: list, metadata: dict) -> dict:
    """Zeroed, schema-valid cost estimate. PLACEHOLDER (TODO T2.2).

    confidence_overall is 0.0 on purpose: with no real costing, a claim must not
    be eligible for the green (auto-resolve) lane.
    """
    return {
        "total_eur": 0.0,
        "p25_eur": 0.0,
        "p75_eur": 0.0,
        "breakdown": {"mano_obra": 0.0, "piezas": 0.0, "materiales": 0.0, "iva": 0.0},
        "confidence_overall": 0.0,
        "currency": "EUR",
        "iva_included": True,
        "parts_lookup_missing": [],
    }


# ── Production default components (lazy, untested offline) ───────────

def _default_damage_detector(model_path: Optional[str] = None) -> Callable:
    """Return a damage detector backed by the trained YOLO seg model.

    Lazily loads the model; raises a clear FileNotFoundError if the weights are
    not present (they live on the remote GPU box until T0.1 completes).
    """
    default_path = PROJECT_ROOT / "models" / "baseline_v1.0" / "best.pt"
    path = Path(model_path) if model_path else default_path
    if not path.exists():
        raise FileNotFoundError(
            f"Damage model not found: {path}. Train it or fetch best.pt from the "
            f"remote GPU run (T0.1), or inject a damage_detector for testing."
        )
    from ultralytics import YOLO  # lazy
    model = YOLO(str(path))

    def detect(image_path: Path) -> list:
        from predict import run_inference  # reuse existing inference mapping
        report = run_inference(model, Path(image_path))
        return [
            {"type": d["class"], "confidence": d["confidence"], "area_pct": d.get("area_pct")}
            for d in report.get("damages", [])
        ]

    return detect


# ── Orchestrator ─────────────────────────────────────────────────────

def assess_claim(
    claim_id: str,
    image_paths: list,
    metadata: dict,
    *,
    damage_detector: Optional[Callable] = None,
    quality_vehicle_detector: Optional[Callable] = None,
    estimator: Optional[Callable] = None,
    model_version: Optional[dict] = None,
    rules: Optional[dict] = None,
    quality_config: Optional[dict] = None,
    audit_config: Optional[dict] = None,
    log_dir: Optional[Path] = None,
    schema: Optional[dict] = None,
    write_audit: bool = True,
) -> dict:
    """Run the full claim assessment pipeline and return the validated output.

    Args:
        claim_id: Claim id (no PII).
        image_paths: List of image paths for the claim.
        metadata: Claim metadata (valor_vehiculo_estimado, siniestros_12m, ...).
        damage_detector: (image_path) -> list of raw damage dicts. Defaults to
            the real YOLO model; inject a fake to run offline.
        quality_vehicle_detector: Injected into the quality gate (offline tests).
        estimator: (damages, metadata) -> estimacion dict. Defaults to the
            zeroed placeholder (TODO T2.2).
        model_version / rules / quality_config / audit_config / schema: optional
            overrides, loaded from disk if None.
        log_dir: Audit log directory (defaults to configs value).
        write_audit: If True (default), append the audit line.

    Returns:
        The schema-validated output dict.

    Raises:
        ValueError: if image_paths is empty.
    """
    if not image_paths:
        raise ValueError("assess_claim requires at least one image.")

    rules = rules if rules is not None else triage.load_rules()
    quality_config = quality_config if quality_config is not None else quality_gate.load_config()
    audit_config = audit_config if audit_config is not None else audit_log.load_config()
    estimator = estimator or _placeholder_estimacion
    damage_detector = damage_detector or _default_damage_detector()
    model_version = model_version or {"damage_model": "baseline_v1.0", "parts_model": "parts_seg_v1.0"}

    t0 = time.perf_counter()

    # ── Per image: hash + quality gate + (if valid) damage inference ──
    input_hashes = []
    per_image_quality = []
    per_image_damages = []
    for img_path in image_paths:
        img_path = Path(img_path)
        image_hash = audit_log.hash_image(img_path)
        input_hashes.append(image_hash)

        verdict = quality_gate.assess_quality(
            img_path, config=quality_config,
            vehicle_detector=quality_vehicle_detector,
            strip_exif=True,
        )
        per_image_quality.append({
            "image_hash": image_hash,
            "valid": verdict["valid"],
            "problems": verdict["problems"],
            "scores": verdict.get("scores", {}),
        })

        if verdict["valid"]:
            raw = damage_detector(img_path)
            per_image_damages.append((image_hash, raw))

    quality = {
        "valid": any(q["valid"] for q in per_image_quality),
        "per_image": per_image_quality,
    }

    # ── Aggregation (placeholder) + estimate (placeholder) ──
    damages = _aggregate(per_image_damages)
    estimacion = estimator(damages, metadata)
    alerts = []  # TODO T2.4: real alert detection

    zones_summary = {}
    for d in damages:
        zones_summary[d["zone"]] = zones_summary.get(d["zone"], 0) + 1

    # ── Deterministic triage ──
    report = {"quality": quality, "damages": damages, "estimacion": estimacion, "alerts": alerts}
    lane, rule_id, reason = triage.assign_lane(report, metadata, rules)
    next_action = (rules.get("next_actions", {}) or {}).get(lane, "Revisión manual.")

    processing_time_ms = int((time.perf_counter() - t0) * 1000)
    audit = {
        "input_hashes": input_hashes,
        "processing_time_ms": processing_time_ms,
        "pipeline_version": PIPELINE_VERSION,
        "rules_versions": {
            "lane_rules": rules.get("version", "unknown"),
            "quality_gate": quality_config.get("version", "unknown"),
        },
    }

    output = output_builder.build_output(
        claim_id=claim_id,
        model_version=model_version,
        quality=quality,
        damages=damages,
        estimacion=estimacion,
        alerts=alerts,
        zones_summary=zones_summary,
        lane=lane,
        lane_rule_id=rule_id,
        lane_reason=reason,
        next_action=next_action,
        audit=audit,
        schema=schema,
    )

    if write_audit:
        audit_log.log_from_output(output, log_dir=log_dir, config=audit_config)

    log.info("claim %s -> lane=%s rule=%s n_damages=%d", claim_id, lane, rule_id, len(damages))
    return output


# ── CLI ──────────────────────────────────────────────────────────────

def _find_images(source: Path) -> list:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Assess a parking-damage claim end to end.")
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--images", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--metadata", type=Path, required=True, help="Claim metadata JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON output here.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Override audit log directory.")
    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    images = _find_images(args.images)
    if not images:
        parser.error(f"No images found in {args.images}")

    output = assess_claim(args.claim_id, images, metadata, log_dir=args.log_dir)

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        log.info("Output written to %s", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
