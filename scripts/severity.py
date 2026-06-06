#!/usr/bin/env python3
"""
severity.py — Economic severity of a damage (T2.3).

Replaces the naive "severity by % of image area" heuristic that lived inline in
predict.py. The authoritative severity is computed per damage as:

    final = max(severidad_visual, severidad_económica)

where
    severidad_visual    = lookup in business_rules/severity_matrix.yaml by
                          (part_category, damage_type, extension)
    severidad_económica = derived from the cost estimate (€ thresholds)

then escalation rules may raise it (e.g. tech headlights → severo; bodywork
cracks → severo + structural suspicion, which feeds the red triage lane).

`preliminary_visual_severity` is the small, config-driven replacement for the
magic-number flag that predict.py shows in standalone inference — explicitly NOT
authoritative.

Public API
----------
    load_severity_matrix(path=None) -> dict
    compute_severity(damage, cost_estimate=None, matrix=None) -> dict
    preliminary_visual_severity(image_area_pct, matrix=None) -> str
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("severity")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "business_rules" / "severity_matrix.yaml"

_RANK = {"leve": 0, "moderado": 1, "severo": 2}
_INV = {0: "leve", 1: "moderado", 2: "severo"}
_DAMAGE_TYPE_TO_MATRIX = {"broken_light": "broken"}
# ESC-3: technological headlights are expensive even with small damage.
_TECH_LIGHTS = {"xenon", "led", "matrix"}


def load_severity_matrix(path: Optional[Path] = None) -> dict:
    import yaml

    matrix_path = Path(path) if path else DEFAULT_MATRIX
    if not matrix_path.exists():
        raise FileNotFoundError(f"Severity matrix not found: {matrix_path}")
    with open(matrix_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _max_severity(a: str, b: str) -> str:
    return _INV[max(_RANK[a], _RANK[b])]


def _matrix_severity(part_category: str, damage_type: str, extension: str, matrix: dict):
    """Return (severity, catalogued) from the matrix, with graceful fallbacks.

    catalogued is False whenever the exact combination was not present (the
    caller should then raise a 'combinación no catalogada' alert); the default
    severity in that case is the conservative 'moderado'.
    """
    table = matrix["severity"]
    cat = part_category if part_category in table else "unknown"
    catalogued = cat == part_category

    damages_map = table[cat]
    dkey = _DAMAGE_TYPE_TO_MATRIX.get(damage_type, damage_type)
    sub = damages_map.get(dkey)
    if sub is None:
        sub = damages_map.get("any")
        catalogued = False
    if sub is None:
        return "moderado", False

    sev = sub.get(extension)
    if sev is None:
        sev = sub.get("any")
        if extension not in (None, "any"):
            catalogued = False
    if sev is None:
        return "moderado", False

    return sev, catalogued


def _cost_severity(total_eur, matrix: dict) -> str:
    th = matrix.get("cost_severity_thresholds", {}) or {}
    if total_eur is None:
        return "leve"  # unknown cost must not raise severity on its own
    if total_eur <= th.get("leve_max_eur", 400):
        return "leve"
    if total_eur <= th.get("moderado_max_eur", 900):
        return "moderado"
    return "severo"


def _resolve_category(damage: dict, matrix: dict) -> str:
    cat = damage.get("part_category")
    if cat and cat != "unknown":
        return cat
    mapped = matrix.get("part_to_category", {}).get(damage.get("part", ""))
    return mapped or cat or "unknown"


def compute_severity(damage: dict, cost_estimate: Optional[dict] = None,
                     matrix: Optional[dict] = None) -> dict:
    """Compute the authoritative economic severity of one consolidated damage.

    Args:
        damage: consolidated damage (part_category or part, type, extension,
            optionally tech, structural_suspicion).
        cost_estimate: the estimacion dict (uses total_eur); optional.
        matrix: pre-loaded severity matrix; loaded if None.

    Returns:
        {severity, structural_suspicion, matrix_severity, cost_severity,
         catalogued, escalations}
    """
    matrix = matrix if matrix is not None else load_severity_matrix()

    category = _resolve_category(damage, matrix)
    dtype = damage.get("type", "")
    extension = damage.get("extension", "any")

    m_sev, catalogued = _matrix_severity(category, dtype, extension, matrix)
    c_sev = _cost_severity((cost_estimate or {}).get("total_eur"), matrix)
    final = _max_severity(m_sev, c_sev)

    escalations = []
    structural = bool(damage.get("structural_suspicion", False))

    # ESC-3: technological headlights are expensive even with minor damage.
    if category == "light_assembly" and damage.get("tech") in _TECH_LIGHTS:
        final = "severo"
        escalations.append("ESC-3")

    # Bodywork crack = suspected structural damage (matrix comment) → red lane.
    dkey = _DAMAGE_TYPE_TO_MATRIX.get(dtype, dtype)
    if category == "body_panel" and dkey == "crack":
        structural = True
        final = "severo"
        escalations.append("ESC-2")

    if structural and final == "leve":
        final = "moderado"  # a structurally-suspected damage is never "leve"

    return {
        "severity": final,
        "structural_suspicion": structural,
        "matrix_severity": m_sev,
        "cost_severity": c_sev,
        "catalogued": catalogued,
        "escalations": escalations,
    }


def preliminary_visual_severity(image_area_pct: float, matrix: Optional[dict] = None) -> str:
    """Quick visual-only severity flag for predict.py standalone (NOT authoritative).

    Config-driven replacement for the old inline magic numbers. Returns the
    Spanish capitalised labels predict.py already used ("Leve"/"Moderado"/"Severo").
    """
    matrix = matrix if matrix is not None else load_severity_matrix()
    th = matrix.get("preliminary_visual_thresholds", {}) or {}
    if image_area_pct < th.get("leve_max_pct", 2.0):
        return "Leve"
    if image_area_pct < th.get("moderado_max_pct", 10.0):
        return "Moderado"
    return "Severo"
