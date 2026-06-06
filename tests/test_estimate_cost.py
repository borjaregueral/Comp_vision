"""
Tests for scripts/estimate_cost.py (T2.2).

Covers the plan cases (Seat Ibiza bumper scratch -> reasonable cost; unknown part
-> low confidence) plus replace pricing, OEM/aftermarket selection, the P25-P75
range ordering, province fallback and the empty-damages base case.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import estimate_cost as ec  # noqa: E402


def _damage(part_category, dtype, extension, part="front_bumper"):
    return {"part_category": part_category, "type": dtype, "extension": extension, "part": part}


def _meta(marca="Seat", modelo="Ibiza", anio=2019, valor=12000):
    return {"marca": marca, "modelo": modelo, "anio": anio, "valor_vehiculo_estimado": valor}


# ── Plan case: scratch on a Seat Ibiza bumper ─────────────────────────

def test_scratch_bumper_seat_ibiza_reasonable():
    damages = [_damage("plastic_panel", "scratch", "medium")]
    est = ec.estimate_repair_cost(damages, _meta(), province="Zaragoza")
    # repaint_only → no part replacement.
    assert est["breakdown"]["piezas"] == 0.0
    assert est["parts_lookup_missing"] == []
    assert est["confidence_label"] == "high"
    assert est["province_used"] == "Zaragoza"
    # A bumper repaint is a low-hundreds-of-euros job, not thousands.
    assert 80.0 < est["total_eur"] < 200.0
    assert est["p25_eur"] <= est["total_eur"] <= est["p75_eur"]


# ── Replacement pricing + OEM/aftermarket selection ──────────────────

def test_replace_uses_aftermarket_for_old_cheap_car():
    damages = [_damage("plastic_panel", "dent", "large")]  # decision: replace
    est = ec.estimate_repair_cost(damages, _meta(anio=2019, valor=12000), province="Zaragoza")
    # Seat Ibiza front_bumper aftermarket = 180 (old, cheap, standard tech).
    assert est["breakdown"]["piezas"] == 180.0
    assert est["confidence_label"] == "high"
    assert est["total_eur"] > 300.0


def test_replace_uses_oem_for_new_car():
    damages = [_damage("plastic_panel", "dent", "large")]
    est = ec.estimate_repair_cost(damages, _meta(anio=2024, valor=12000), province="Zaragoza")
    # <=3 years old → OEM (320).
    assert est["breakdown"]["piezas"] == 320.0


def test_replace_uses_oem_for_premium_value():
    damages = [_damage("plastic_panel", "dent", "large")]
    est = ec.estimate_repair_cost(damages, _meta(anio=2019, valor=35000), province="Zaragoza")
    assert est["breakdown"]["piezas"] == 320.0  # value > 30000 → OEM


# ── Unknown part → low confidence (derives to amber) ─────────────────

def test_unknown_part_low_confidence():
    damages = [_damage("plastic_panel", "dent", "large", part="front_bumper")]
    est = ec.estimate_repair_cost(damages, _meta(marca="Tesla", modelo="Model Y"), province="Zaragoza")
    assert est["confidence_label"] == "low"
    assert est["confidence"] == 0.40
    assert "front_bumper" in est["parts_lookup_missing"]
    # Uses the fallback price for an uncatalogued front_bumper.
    assert est["breakdown"]["piezas"] == 400.0


# ── Baremo fallback → medium confidence ──────────────────────────────

def test_unknown_part_category_medium_confidence():
    # 'glass' is not in the baremo → falls back to 'unknown' (no part replace here).
    damages = [_damage("glass", "scratch", "small", part="windshield")]
    est = ec.estimate_repair_cost(damages, _meta(), province="Zaragoza")
    assert est["confidence_label"] == "medium"
    assert est["parts_lookup_missing"] == []


# ── Range, province fallback, empty ──────────────────────────────────

def test_range_ordering_multiple_damages():
    damages = [
        _damage("plastic_panel", "scratch", "small"),
        _damage("body_panel", "dent", "medium", part="front_left_door"),
    ]
    est = ec.estimate_repair_cost(damages, _meta(), province="Madrid")
    assert est["p25_eur"] <= est["total_eur"] <= est["p75_eur"]
    assert est["total_eur"] > 0


def test_unknown_province_uses_default():
    damages = [_damage("plastic_panel", "scratch", "small")]
    est = ec.estimate_repair_cost(damages, _meta(), province="Teruel")
    assert est["province_used"] == "default"


def test_empty_damages_is_zero():
    est = ec.estimate_repair_cost([], _meta(), province="Zaragoza")
    assert est["total_eur"] == 0.0
    assert est["p25_eur"] == 0.0 and est["p75_eur"] == 0.0
    assert est["breakdown"] == {"mano_obra": 0.0, "piezas": 0.0, "materiales": 0.0, "iva": 0.0}
    assert est["parts_lookup_missing"] == []


def test_output_shape_matches_estimacion_contract():
    est = ec.estimate_repair_cost([_damage("plastic_panel", "scratch", "small")], _meta(), "Zaragoza")
    for key in ("total_eur", "p25_eur", "p75_eur", "breakdown", "confidence",
                "currency", "iva_included", "province_used", "parts_lookup_missing"):
        assert key in est
    assert est["currency"] == "EUR"
    assert 0.0 <= est["confidence"] <= 1.0
