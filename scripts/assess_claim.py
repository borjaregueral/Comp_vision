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
import alerts
import audit_log
import claim_aggregator
import estimate_cost
import output_builder
import quality_gate
import severity
import triage

log = logging.getLogger("assess_claim")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = "assess_claim/0.2.0"  # Sprint 2: real cost/severity/alerts/aggregation

_DAMAGE_TYPES = {"dent", "scratch", "crack", "broken_light", "paint_chip", "puncture"}


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
            {"type": d["class"], "confidence": d["confidence"], "area_pct": d.get("area_pct"),
             "bbox": d.get("bbox"), "mask_polygon": d.get("mask_polygon")}
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
    parts_model: Optional[object] = None,
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
        estimator: optional (damages, metadata, province) -> cost dict override;
            defaults to estimate_cost.estimate_repair_cost with the real tables.
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
    damage_detector = damage_detector or _default_damage_detector()
    model_version = model_version or {"damage_model": "baseline_v1.0", "parts_model": "parts_seg_v1.0"}

    province = metadata.get("provincia")
    severity_matrix = severity.load_severity_matrix()
    alerts_config = alerts.load_config()

    # Parts/zone localization (T4.5): if a parts model is supplied, each damage is
    # assigned a zone + part (carparts-seg overlap) -> part_category for real costing.
    # Without it, damages keep zone="unknown" and the unknown-part fallback applies.
    part_to_category = severity_matrix.get("part_to_category", {}) or {}
    parts_cfg = None
    if parts_model is not None:
        import localize
        parts_cfg = localize.load_parts_config()

    # Cost estimator: the real estimate_cost (reference tables loaded ONCE) unless
    # a custom estimator(damages, metadata, province) is injected.
    if estimator is None:
        _baremo = estimate_cost.load_baremo()
        _precios = estimate_cost.load_precios()
        _piezas = estimate_cost.load_piezas()
        _est_cfg = estimate_cost.load_estimation_config()
        pricing_versions = {
            "baremo_horas": _baremo.get("version", "unknown"),
            "precios_taller": _precios.get("version", "unknown"),
            "piezas": _piezas.get("version", "unknown"),
        }
    else:
        _baremo = _precios = _piezas = _est_cfg = None
        pricing_versions = {"baremo_horas": "injected", "precios_taller": "injected", "piezas": "injected"}

    def _estimate(dmgs):
        if estimator is not None:
            return estimator(dmgs, metadata, province)
        return estimate_cost.estimate_repair_cost(
            dmgs, metadata, province,
            baremo=_baremo, precios=_precios, piezas=_piezas, config=_est_cfg,
        )

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
            raw = [d for d in damage_detector(img_path) if d.get("type") in _DAMAGE_TYPES]
            if parts_cfg is not None and raw:  # parts_cfg set iff parts_model provided
                import localize
                localize.enrich_report_with_zones({"damages": raw}, img_path, parts_model, parts_cfg)
                for d in raw:
                    matched = d.get("matched_part")
                    d["part"] = matched or "unknown"
                    d["part_category"] = part_to_category.get(matched, "unknown")
            per_image_damages.append({"image_hash": image_hash, "damages": raw})

    quality = {
        "valid": any(q["valid"] for q in per_image_quality),
        "per_image": per_image_quality,
    }

    # ── Multi-view aggregation (T2.5): dedup same damage across photos ──
    aggregated = claim_aggregator.aggregate_claim(per_image_damages)
    damages = aggregated["damages"]

    # ── Per-damage economic severity (T2.3) using its own cost (T2.2) ──
    for damage in damages:
        damage_cost = _estimate([damage])
        sev = severity.compute_severity(damage, damage_cost, matrix=severity_matrix)
        damage["severity"] = sev["severity"]
        damage["structural_suspicion"] = bool(damage.get("structural_suspicion")) or sev["structural_suspicion"]

    # ── Claim-level cost estimate over UNIQUE consolidated damages (T2.2) ──
    est = _estimate(damages)
    estimacion = {
        "total_eur": est["total_eur"],
        "p25_eur": est["p25_eur"],
        "p75_eur": est["p75_eur"],
        "breakdown": est["breakdown"],
        "confidence_overall": est["confidence"],
        "currency": "EUR",
        "iva_included": True,
        "province_used": est.get("province_used", "default"),
        "parts_lookup_missing": est.get("parts_lookup_missing", []),
    }

    # ── Alerts (T2.4). crops not wired yet → preexisting heuristic skipped. ──
    claim_alerts = alerts.detect_alerts(
        damages, metadata, crops=None, image_paths=image_paths, config=alerts_config
    )

    zones_summary = {}
    for d in damages:
        zones_summary[d["zone"]] = zones_summary.get(d["zone"], 0) + 1

    # ── Deterministic triage (now on real cost / severity / alerts) ──
    report = {"quality": quality, "damages": damages, "estimacion": estimacion, "alerts": claim_alerts}
    lane, rule_id, reason = triage.assign_lane(report, metadata, rules)
    next_action = (rules.get("next_actions", {}) or {}).get(lane, "Revisión manual.")

    processing_time_ms = int((time.perf_counter() - t0) * 1000)
    audit = {
        "input_hashes": input_hashes,
        "processing_time_ms": processing_time_ms,
        "pipeline_version": PIPELINE_VERSION,
        "rules_versions": {
            "lane_rules": rules.get("version", "unknown"),
            "severity_matrix": severity_matrix.get("version", "unknown"),
            **pricing_versions,
        },
    }

    output = output_builder.build_output(
        claim_id=claim_id,
        model_version=model_version,
        quality=quality,
        damages=damages,
        estimacion=estimacion,
        alerts=claim_alerts,
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
    parser.add_argument("--model", type=Path, default=None,
                        help="Damage model .pt (default: models/baseline_v1.0/best.pt).")
    parser.add_argument("--parts-model", type=Path, default=None,
                        help="Parts seg model .pt for zone/part assignment "
                             "(e.g. runs/parts_seg/train/weights/best.pt). Without it, zone=unknown.")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON output here.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Override audit log directory.")
    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    images = _find_images(args.images)
    if not images:
        parser.error(f"No images found in {args.images}")

    detector = _default_damage_detector(str(args.model)) if args.model else None
    parts_model = None
    if args.parts_model:
        from ultralytics import YOLO
        parts_model = YOLO(str(args.parts_model))
    model_version = {
        "damage_model": args.model.parent.name if args.model else "default",
        "parts_model": "carparts-seg" if args.parts_model else "none",
    }
    output = assess_claim(args.claim_id, images, metadata, damage_detector=detector,
                          parts_model=parts_model, model_version=model_version, log_dir=args.log_dir)

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        log.info("Output written to %s", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
