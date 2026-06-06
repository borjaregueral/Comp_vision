#!/usr/bin/env python3
"""
estimate_cost.py — Repair cost estimation in euros (T2.2).

Turns consolidated damages into a cost estimate using the validated reference
tables (baremo_horas, precios_taller, piezas). This is the module that justifies
the ROI: the primary business metric is MAE in euros, not mAP.

Cost model (per the baremo header):
    mano_obra  = chapa_h · €/h_chapa + pintura_h · €/h_pintura      (by province)
    materiales = (pintura_h · €/h_pintura) · paint_materials_pct
    piezas     = part price (only if baremo decision == "replace")
    subtotal   = mano_obra + materiales + piezas
    total      = subtotal · (1 + IVA)

It returns a P25–P75 RANGE, not just a point. The range is a heuristic band
(±labor_uncertainty on labour + the OEM/aftermarket spread on parts), NOT a
calibrated statistical percentile — recalibrate against real paid amounts in
Sprint 3. When liquidating conservatively, use p75_eur.

If a part needed for a replacement is not catalogued, the estimate uses a
fallback price, lists it in parts_lookup_missing and drops confidence to "low",
which deterministically keeps the claim out of the green lane (→ amber).

Public API
----------
    estimate_repair_cost(damages, vehicle_metadata, province, ...) -> dict
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("estimate_cost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Map the detector's class to the baremo's generic damage vocabulary.
_DAMAGE_TYPE_TO_BAREMO = {"broken_light": "broken"}
# Common brand spelling normalisations for the parts catalog lookup.
_BRAND_ALIASES = {"vw": "volkswagen"}


# ── Loaders ──────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_baremo(path=None) -> dict:
    return _load_yaml(Path(path) if path else CONFIG_DIR / "baremo_horas.yaml")


def load_precios(path=None) -> dict:
    return _load_yaml(Path(path) if path else CONFIG_DIR / "precios_taller.yaml")


def load_piezas(path=None) -> dict:
    return _load_yaml(Path(path) if path else CONFIG_DIR / "piezas.yaml")


def load_estimation_config(path=None) -> dict:
    return _load_yaml(Path(path) if path else CONFIG_DIR / "estimation.yaml")


# ── Resolvers ────────────────────────────────────────────────────────

def _resolve_rates(province: Optional[str], precios: dict):
    """Return (chapa_eur_h, pintura_eur_h, province_used) for a province."""
    by_prov = precios.get("by_province", {}) or {}
    if province:
        target = province.strip().lower()
        for name, rates in by_prov.items():
            if name.lower() == target:
                return rates["chapa_eur_h"], rates["pintura_eur_h"], name
    default = precios["default"]
    return default["chapa_eur_h"], default["pintura_eur_h"], "default"


def _resolve_baremo(part_category: str, damage_type: str, extension: str, baremo: dict):
    """Resolve a baremo leaf, returning (leaf, exact_match).

    Falls back gracefully: unknown category → 'unknown'; missing damage type or
    extension → the 'any' bucket. exact_match is False whenever any fallback was
    used (this lowers the overall confidence).
    """
    table = baremo["baremo"]
    exact = True

    cat = part_category if part_category in table else "unknown"
    if cat != part_category:
        exact = False
    damages_map = table[cat]

    dkey = _DAMAGE_TYPE_TO_BAREMO.get(damage_type, damage_type)
    if dkey in damages_map:
        ext_map = damages_map[dkey]
    elif "any" in damages_map:
        ext_map = damages_map["any"]
        exact = False
    else:
        # Last resort: first available damage entry.
        ext_map = next(iter(damages_map.values()))
        exact = False

    if extension in ext_map:
        leaf = ext_map[extension]
    elif "any" in ext_map:
        leaf = ext_map["any"]
        exact = exact and (extension == "any")
    else:
        leaf = next(iter(ext_map.values()))
        exact = False

    return leaf, exact


def _match_key(value: Optional[str], mapping: dict, aliases: Optional[dict] = None):
    """Case-insensitive key match (with optional alias map). Returns the real key or None."""
    if not value:
        return None
    key = value.strip().lower()
    if aliases:
        key = aliases.get(key, key)
    for real in mapping:
        if str(real).lower() == key:
            return real
    return None


def _use_oem(spec: dict, vehicle_metadata: dict, cfg: dict) -> bool:
    """Decide OEM vs aftermarket per the selection policy (mirrors piezas.yaml)."""
    oem_cfg = cfg.get("oem_selection", {})
    if spec.get("aftermarket_eur") is None:
        return True  # no alternative
    anio = vehicle_metadata.get("anio")
    if anio is not None:
        age = cfg.get("reference_year", 2026) - anio
        if age <= oem_cfg.get("age_years_max", 3):
            return True
    value = vehicle_metadata.get("valor_vehiculo_estimado")
    if value is not None and value > oem_cfg.get("value_min_eur", 30000):
        return True
    if spec.get("tech") in set(oem_cfg.get("tech_oem_only", [])):
        return True
    return False


def _resolve_part_price(damage: dict, vehicle_metadata: dict, piezas: dict, cfg: dict):
    """Return (low, point, high, missing) for a part replacement.

    Looks up brand→model→part_family in the catalog; if absent, uses
    fallback_prices (missing=True) and finally a generic fallback.
    """
    part_family = damage.get("part", "unknown")
    catalog = piezas.get("catalog", {}) or {}

    brand_key = _match_key(vehicle_metadata.get("marca"), catalog, _BRAND_ALIASES)
    spec = None
    if brand_key is not None:
        models = catalog[brand_key]
        model_key = _match_key(vehicle_metadata.get("modelo"), models)
        if model_key is not None:
            spec = models[model_key].get(part_family)

    if spec is not None:
        oem = spec["oem_eur"]
        after = spec.get("aftermarket_eur")
        point = oem if _use_oem(spec, vehicle_metadata, cfg) else (after if after is not None else oem)
        low = after if after is not None else oem
        high = oem
        return float(low), float(point), float(high), False

    # Not catalogued → fallback price; flag as missing (confidence -> low).
    fallback = (piezas.get("fallback_prices", {}) or {}).get(part_family)
    price = float(fallback if fallback is not None else cfg.get("generic_part_fallback_eur", 400))
    return price, price, price, True


# ── Main API ─────────────────────────────────────────────────────────

def estimate_repair_cost(
    damages: list,
    vehicle_metadata: dict,
    province: Optional[str] = None,
    *,
    baremo: Optional[dict] = None,
    precios: Optional[dict] = None,
    piezas: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """Estimate repair cost (EUR) for a list of consolidated damages.

    Args:
        damages: consolidated damages; each uses part_category, type, extension,
            and part (for pricing replacements).
        vehicle_metadata: marca, modelo, anio, valor_vehiculo_estimado, ...
        province: province name (case-insensitive; 'default' tariff if unknown).

    Returns:
        {total_eur, p25_eur, p75_eur, breakdown{mano_obra,piezas,materiales,iva},
         confidence (0-1 == schema confidence_overall), confidence_label,
         currency, iva_included, province_used, parts_lookup_missing}
    """
    baremo = baremo if baremo is not None else load_baremo()
    precios = precios if precios is not None else load_precios()
    piezas = piezas if piezas is not None else load_piezas()
    config = config if config is not None else load_estimation_config()

    chapa_rate, pintura_rate, province_used = _resolve_rates(province, precios)
    band = config.get("labor_uncertainty_band_pct", 20.0) / 100.0
    paint_pct = baremo["paint_materials"]["pct_of_pintura_h_cost"] / 100.0
    vat_rate = precios["vat"]["rate_pct"] / 100.0

    mano_obra_sum = materiales_sum = piezas_sum = 0.0
    sub_point = sub_low = sub_high = 0.0
    parts_lookup_missing = []
    baremo_fallback_used = False

    for d in damages:
        leaf, exact = _resolve_baremo(
            d.get("part_category", "unknown"), d.get("type", ""), d.get("extension", "any"), baremo
        )
        if not exact:
            baremo_fallback_used = True

        chapa_h = leaf["chapa_h"]
        pintura_h = leaf["pintura_h"]
        decision = leaf["decision"]

        pintura_labor = pintura_h * pintura_rate
        mano_obra = chapa_h * chapa_rate + pintura_h * pintura_rate
        materiales = pintura_labor * paint_pct
        labor_block = mano_obra + materiales

        if decision == "replace":
            p_low, p_point, p_high, missing = _resolve_part_price(d, vehicle_metadata, piezas, config)
            if missing:
                parts_lookup_missing.append(d.get("part", "unknown"))
        else:
            p_low = p_point = p_high = 0.0

        mano_obra_sum += mano_obra
        materiales_sum += materiales
        piezas_sum += p_point
        sub_point += labor_block + p_point
        sub_low += labor_block * (1 - band) + p_low
        sub_high += labor_block * (1 + band) + p_high

    if parts_lookup_missing:
        label = "low"
    elif baremo_fallback_used:
        label = "medium"
    else:
        label = "high"
    confidence = config["confidence_levels"][label]

    def _eur(x):
        return round(x * (1 + vat_rate), 2)

    return {
        "total_eur": _eur(sub_point),
        "p25_eur": _eur(sub_low),
        "p75_eur": _eur(sub_high),
        "breakdown": {
            "mano_obra": round(mano_obra_sum, 2),
            "piezas": round(piezas_sum, 2),
            "materiales": round(materiales_sum, 2),
            "iva": round(sub_point * vat_rate, 2),
        },
        "confidence": confidence,            # 0-1, maps to estimacion.confidence_overall
        "confidence_label": label,           # low | medium | high
        "currency": "EUR",
        "iva_included": True,
        "province_used": province_used,
        "parts_lookup_missing": parts_lookup_missing,
    }
