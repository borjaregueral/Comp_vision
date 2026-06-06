#!/usr/bin/env python3
"""
alerts.py — Heuristic alert detection (T2.4).

Detects preexisting damage, declared-vs-detected part mismatches, multiple
unrelated damages (possible fraud) and image manipulation. All detectors are
HEURISTIC (no ML); the preexisting and manipulation ones are explicit v1
placeholders (TODO: trained classifier T4.4 / JPEG double-compression + ELA).

Each alert is {id, severity (info|warning|critical), description, evidence} —
matching the alerts items of schemas/inference_output_v1.json. Critical alerts
force the red lane (triage rule ROJO-3); v1 heuristics deliberately emit only
'warning' to avoid false reds.

Public API
----------
    load_config(path=None) -> dict
    alert_preexisting_damage(crop, config=None) -> dict | None
    alert_part_declaration_mismatch(damages, descripcion, config=None) -> dict | None
    alert_multiple_unrelated_damages(damages, config=None) -> dict | None
    alert_image_manipulation(image_paths, config=None) -> dict | None
    detect_alerts(damages, vehicle_metadata, *, crops=None, image_paths=None, config=None) -> list
"""

import logging
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("alerts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "alerts.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Alerts config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize(text: str) -> str:
    """Lowercase + strip accents (so 'capó' == 'capo')."""
    norm = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in norm if not unicodedata.combining(c))


# ── Detectors ────────────────────────────────────────────────────────

def alert_preexisting_damage(crop, config: Optional[dict] = None) -> Optional[dict]:
    """v1 heuristic: rust/brown HSV fraction in the damage crop.

    PLACEHOLDER — TODO v2: a trained fresh-vs-preexisting classifier (T4.4).
    Returns None if disabled or no crop given.
    """
    config = config if config is not None else load_config()
    cfg = config.get("preexisting_damage", {})
    if not cfg.get("enabled", False) or crop is None:
        return None

    import cv2
    crop = np.asarray(crop)
    if crop.size == 0 or crop.ndim != 3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.get("rust_hsv_lower", [5, 50, 20]))
    upper = np.array(cfg.get("rust_hsv_upper", [25, 255, 200]))
    mask = cv2.inRange(hsv, lower, upper)
    fraction = float(np.count_nonzero(mask)) / mask.size

    if fraction >= cfg.get("min_rust_fraction", 0.15):
        return {
            "id": "preexisting_damage_suspected",
            "severity": cfg.get("severity", "warning"),
            "description": "Posible daño preexistente (óxido/suciedad en el área del daño).",
            "evidence": {"rust_fraction": round(fraction, 3)},
        }
    return None


def alert_part_declaration_mismatch(damages, descripcion, config: Optional[dict] = None) -> Optional[dict]:
    """Flag when the insured's declared part(s) do not match any detected damage."""
    config = config if config is not None else load_config()
    cfg = config.get("part_declaration_mismatch", {})
    if not cfg.get("enabled", False):
        return None

    desc = _normalize(descripcion)
    if not desc:
        return None  # nothing declared → cannot assess

    keyword_map = cfg.get("keyword_to_part", {})
    declared = set()
    declared_terms = []
    for keyword, parts in keyword_map.items():
        if _normalize(keyword) in desc:
            declared.update(parts)
            declared_terms.append(keyword)
    if not declared:
        return None  # no recognised part term in the description

    detected = set()
    for d in damages or []:
        for key in ("part", "part_category", "zone"):
            val = d.get(key)
            if val and val != "unknown":
                detected.add(val)

    if declared.isdisjoint(detected):
        return {
            "id": "part_declaration_mismatch",
            "severity": cfg.get("severity", "warning"),
            "description": (
                f"La descripción menciona {declared_terms} pero el daño detectado "
                f"no coincide con esa zona."
            ),
            "evidence": {"declared": sorted(declared), "detected": sorted(detected)},
        }
    return None


def alert_multiple_unrelated_damages(damages, config: Optional[dict] = None) -> Optional[dict]:
    """Flag many damages spread over distinct zones with distinct typologies."""
    config = config if config is not None else load_config()
    cfg = config.get("multiple_unrelated_damages", {})
    if not cfg.get("enabled", False):
        return None

    zones = {d.get("zone") for d in (damages or []) if d.get("zone") and d.get("zone") != "unknown"}
    types = {d.get("type") for d in (damages or []) if d.get("type")}
    if len(zones) >= cfg.get("min_distinct_zones", 3) and len(types) >= cfg.get("min_distinct_types", 2):
        return {
            "id": "multiple_unrelated_damages",
            "severity": cfg.get("severity", "warning"),
            "description": "Daños en múltiples zonas no contiguas con tipologías distintas.",
            "evidence": {"zones": sorted(zones), "types": sorted(types)},
        }
    return None


def alert_image_manipulation(image_paths, config: Optional[dict] = None) -> Optional[dict]:
    """PLACEHOLDER — TODO v2: JPEG double-compression + ELA + EXIF consistency.

    Disabled by default; returns None. The id 'image_manipulation' is reserved so
    triage rule ROJO-6 can reference it once this is implemented.
    """
    config = config if config is not None else load_config()
    cfg = config.get("image_manipulation", {})
    if not cfg.get("enabled", False):
        return None
    # TODO v2: real analysis. Until then, even if enabled, emit nothing.
    return None


# ── Orchestrator ─────────────────────────────────────────────────────

def detect_alerts(
    damages,
    vehicle_metadata: dict,
    *,
    crops: Optional[dict] = None,
    image_paths=None,
    config: Optional[dict] = None,
) -> list:
    """Run all detectors and return the list of triggered alerts.

    Args:
        damages: consolidated damages.
        vehicle_metadata: claim metadata (uses descripcion_asegurado).
        crops: optional {damage_id: crop_ndarray} for the preexisting heuristic.
        image_paths: optional image paths (for the manipulation placeholder).
    """
    config = config if config is not None else load_config()
    alerts = []

    for damage_id, crop in (crops or {}).items():
        alert = alert_preexisting_damage(crop, config)
        if alert:
            alert["evidence"] = {**alert.get("evidence", {}), "damage_id": damage_id}
            alerts.append(alert)

    mismatch = alert_part_declaration_mismatch(
        damages, vehicle_metadata.get("descripcion_asegurado", ""), config
    )
    if mismatch:
        alerts.append(mismatch)

    multiple = alert_multiple_unrelated_damages(damages, config)
    if multiple:
        alerts.append(multiple)

    manipulation = alert_image_manipulation(image_paths, config)
    if manipulation:
        alerts.append(manipulation)

    return alerts
