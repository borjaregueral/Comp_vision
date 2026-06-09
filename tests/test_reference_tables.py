"""
Tests for the economic reference tables (T2.1):
configs/baremo_horas.yaml, configs/precios_taller.yaml, configs/piezas.yaml.

Beyond "they parse", these enforce structure, value sanity, the minimum
coverage documented in configs/REFERENCES.md, and cross-consistency with the
output schema (part_category / repair_decision vocabularies). They also assert
the PLACEHOLDER marker is present, so real data cannot be shipped silently
without bumping the version.
"""

import json
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "inference_output_v1.json"

_DAMAGE_TYPES = {"scratch", "dent", "crack", "broken", "paint_chip", "puncture", "any"}
_EXTENSIONS = {"small", "medium", "large", "any"}
_DECISIONS = {"repair", "repaint_only", "replace"}
_REQUIRED_BRANDS = {"SEAT", "Renault", "Peugeot", "Volkswagen", "Toyota", "Ford"}
_MIN_PROVINCES = {"Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga", "Bizkaia"}
_FALLBACK_CORE = {"front_bumper", "back_bumper", "front_left_light", "left_mirror", "wheel"}


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def baremo():
    return _load("baremo_horas.yaml")


@pytest.fixture(scope="module")
def precios():
    return _load("precios_taller.yaml")


@pytest.fixture(scope="module")
def piezas():
    return _load("piezas.yaml")


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ── Common: parse + placeholder marker + versioning ──────────────────

@pytest.mark.parametrize("name", ["baremo_horas.yaml", "precios_taller.yaml", "piezas.yaml"])
def test_table_parses_and_is_versioned(name):
    data = _load(name)
    assert isinstance(data, dict)
    assert isinstance(data.get("version"), str) and data["version"]
    assert isinstance(data.get("effective_date"), str) and data["effective_date"]


@pytest.mark.parametrize("name", ["baremo_horas.yaml", "precios_taller.yaml", "piezas.yaml"])
def test_placeholder_marker_present(name):
    """While values are placeholders, the marker must be present (rule 19)."""
    data = _load(name)
    blob = (data.get("version", "") + " " + str(data.get("status", ""))).upper()
    assert "PLACEHOLDER" in blob or "DRAFT" in blob


# ── baremo_horas.yaml ────────────────────────────────────────────────

def test_baremo_structure_and_values(baremo, schema):
    schema_part_cats = set(
        schema["properties"]["damages"]["items"]["properties"]["part_category"]["enum"]
    )
    schema_decisions = set(
        schema["properties"]["damages"]["items"]["properties"]["repair_decision"]["enum"]
    )
    assert _DECISIONS == schema_decisions  # our decision vocabulary matches the schema

    baremo_map = baremo["baremo"]
    assert baremo_map, "baremo is empty"
    for part_cat, damages in baremo_map.items():
        assert part_cat in schema_part_cats, f"unknown part_category: {part_cat}"
        assert damages, f"{part_cat} has no damage entries"
        for dmg, exts in damages.items():
            assert dmg in _DAMAGE_TYPES, f"{part_cat}/{dmg} bad damage type"
            for ext, leaf in exts.items():
                assert ext in _EXTENSIONS, f"{part_cat}/{dmg}/{ext} bad extension"
                assert _is_number(leaf["chapa_h"]) and leaf["chapa_h"] >= 0
                assert _is_number(leaf["pintura_h"]) and leaf["pintura_h"] >= 0
                assert leaf["decision"] in schema_decisions


def test_baremo_minimum_coverage(baremo):
    baremo_map = baremo["baremo"]
    for cat in ("plastic_panel", "body_panel"):
        assert cat in baremo_map
        for dmg in ("scratch", "dent", "crack"):
            assert dmg in baremo_map[cat], f"{cat} missing {dmg}"
    assert _is_number(baremo["paint_materials"]["pct_of_pintura_h_cost"])
    for op, hours in baremo["additional_operations"].items():
        assert _is_number(hours) and hours >= 0, f"bad additional op {op}"


# ── precios_taller.yaml ──────────────────────────────────────────────

def test_precios_defaults_and_vat(precios):
    assert precios["currency"] == "EUR"
    assert isinstance(precios["iva_included"], bool)
    default = precios["default"]
    assert default["chapa_eur_h"] > 0 and default["pintura_eur_h"] > 0
    assert precios["vat"]["rate_pct"] == 21.0


def test_precios_provinces_coverage_and_consistency(precios):
    provinces = precios["by_province"]
    missing = _MIN_PROVINCES - set(provinces)
    assert not missing, f"missing provinces: {missing}"
    for name, rates in provinces.items():
        assert rates["chapa_eur_h"] > 0, f"{name} bad chapa rate"
        assert rates["pintura_eur_h"] > 0, f"{name} bad pintura rate"
        # Painting is pricier per hour than bodywork (materials/booth).
        assert rates["pintura_eur_h"] >= rates["chapa_eur_h"], f"{name}: pintura < chapa"


def test_precios_modifiers_non_negative(precios):
    mods = precios["modifiers"]
    for key in ("taller_concertado_discount_pct", "premium_vehicle_surcharge_pct",
                "urgent_repair_surcharge_pct"):
        assert _is_number(mods[key]) and mods[key] >= 0


# ── piezas.yaml ──────────────────────────────────────────────────────

def _iter_parts(piezas):
    for brand, models in piezas["catalog"].items():
        for model, parts in models.items():
            for family, spec in parts.items():
                yield brand, model, family, spec


def test_piezas_required_brands_present(piezas):
    brands = set(piezas["catalog"])
    missing = _REQUIRED_BRANDS - brands
    assert not missing, f"missing brands: {missing}"


def test_piezas_entries_valid_and_min_coverage(piezas):
    count = 0
    for brand, model, family, spec in _iter_parts(piezas):
        where = f"{brand}/{model}/{family}"
        yr = spec["year_range"]
        assert isinstance(yr, list) and len(yr) == 2, f"{where} bad year_range"
        assert all(isinstance(v, int) for v in yr) and yr[0] <= yr[1], f"{where} year_range order"
        assert _is_number(spec["oem_eur"]) and spec["oem_eur"] > 0, f"{where} bad oem_eur"
        if "aftermarket_eur" in spec and spec["aftermarket_eur"] is not None:
            assert _is_number(spec["aftermarket_eur"]) and spec["aftermarket_eur"] > 0, where
        assert isinstance(spec["tech"], str) and spec["tech"], f"{where} bad tech"
        if "paint_required" in spec:
            assert isinstance(spec["paint_required"], bool), f"{where} bad paint_required"
        count += 1
    assert count >= 20, f"catalog has only {count} parts; plan requires ~20 top parts"


def test_piezas_selection_policy_and_fallback(piezas):
    policy = piezas["selection_policy"]
    assert policy["default"] in ("oem", "aftermarket")
    assert isinstance(policy["use_oem_if"], list) and policy["use_oem_if"]

    fallback = piezas["fallback_prices"]
    assert _FALLBACK_CORE <= set(fallback), f"fallback missing core families: {_FALLBACK_CORE - set(fallback)}"
    for family, price in fallback.items():
        assert _is_number(price) and price > 0, f"fallback {family} bad price"
