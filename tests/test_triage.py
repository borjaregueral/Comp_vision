"""
Tests for scripts/triage.py (T1.3).

One case per rule (ROJO-1..5, VERDE-1, AMBAR-1), the threshold edge cases, red
precedence over green, and graceful handling when the cost estimate is missing
(the placeholder state before Sprint 2). Triage must be deterministic.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import triage  # noqa: E402

_HASH = "a" * 64
_RULE_ID_RE = re.compile(r"^(VERDE|AMBAR|ROJO)-[0-9]+$")


@pytest.fixture(scope="module")
def rules():
    return triage.load_rules()


def _green_report() -> dict:
    """A report that, with _meta(), satisfies every VERDE-1 condition."""
    return {
        "quality": {"valid": True, "per_image": [{"image_hash": _HASH, "valid": True, "problems": []}]},
        "damages": [{
            "damage_id": "D1", "part": "front_bumper",
            "confidence": 0.90, "structural_suspicion": False,
        }],
        "estimacion": {"total_eur": 300.0, "confidence_overall": 0.90},
        "alerts": [],
    }


def _meta(**over) -> dict:
    m = {"valor_vehiculo_estimado": 12000, "siniestros_12m": 1}
    m.update(over)
    return m


# ── Happy path: green ─────────────────────────────────────────────────

def test_verde(rules):
    lane, rule_id, reason = triage.assign_lane(_green_report(), _meta(), rules)
    assert lane == "verde"
    assert rule_id == "VERDE-1"
    assert reason  # non-empty human-readable reason
    assert _RULE_ID_RE.match(rule_id)


# ── Red rules ─────────────────────────────────────────────────────────

def test_rojo_structural(rules):
    r = _green_report()
    r["damages"][0]["structural_suspicion"] = True
    lane, rule_id, reason = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-1")
    assert "front_bumper" in reason


def test_rojo_cost_over_ceiling(rules):
    r = _green_report()
    r["estimacion"]["total_eur"] = 1600.0  # > 1500
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-2")


def test_rojo_critical_alert(rules):
    r = _green_report()
    r["alerts"] = [{"id": "fraud_suspected", "severity": "critical", "description": "posible fraude"}]
    lane, rule_id, reason = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-3")
    assert "fraude" in reason


def test_rojo_high_value_vehicle(rules):
    lane, rule_id, _ = triage.assign_lane(_green_report(), _meta(valor_vehiculo_estimado=50000), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-4")


def test_rojo_claim_history(rules):
    lane, rule_id, _ = triage.assign_lane(_green_report(), _meta(siniestros_12m=4), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-5")


def test_rojo_invalid_quality_with_manipulation(rules):
    """ROJO-6: invalid quality only forces red when an image-manipulation alert is present."""
    r = _green_report()
    r["quality"] = {"valid": False,
                    "per_image": [{"image_hash": _HASH, "valid": False, "problems": ["blurry"]}]}
    r["alerts"] = [{"id": "image_manipulation", "severity": "warning",
                    "description": "doble compresión JPEG"}]
    lane, rule_id, reason = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("rojo", "ROJO-6")
    assert "blurry" in reason


def test_invalid_quality_without_manipulation_is_amber(rules):
    """Invalid quality WITHOUT manipulation is not ROJO-6 (additional_condition) → amber."""
    r = _green_report()
    r["quality"] = {"valid": False,
                    "per_image": [{"image_hash": _HASH, "valid": False, "problems": ["blurry"]}]}
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert lane == "ambar"


# ── Amber (default) ───────────────────────────────────────────────────

def test_ambar_mid_cost(rules):
    r = _green_report()
    r["estimacion"]["total_eur"] = 1000.0  # 800 <= x <= 1500 → neither green nor red
    lane, rule_id, reason = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("ambar", "AMBAR-1")
    assert "importe<800" in reason  # missing criterion is reported


def test_ambar_low_confidence(rules):
    r = _green_report()
    r["estimacion"]["confidence_overall"] = 0.70
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("ambar", "AMBAR-1")


# ── Threshold edges ───────────────────────────────────────────────────

def test_edge_cost_exactly_green_max_is_not_green(rules):
    r = _green_report()
    r["estimacion"]["total_eur"] = 800.0  # rule is strict < 800
    lane, _, _ = triage.assign_lane(r, _meta(), rules)
    assert lane == "ambar"


def test_edge_cost_exactly_red_min_is_not_red(rules):
    r = _green_report()
    r["estimacion"]["total_eur"] = 1500.0  # rule is strict > 1500
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert lane == "ambar"  # not green (>=800), not red (not >1500)


def test_edge_confidence_exactly_min_is_green(rules):
    r = _green_report()
    r["estimacion"]["confidence_overall"] = 0.85  # >= 0.85
    r["damages"][0]["confidence"] = 0.85
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("verde", "VERDE-1")


# ── Precedence & robustness ───────────────────────────────────────────

def test_red_takes_precedence_over_green(rules):
    """A would-be-green report with a structural suspicion must go red."""
    r = _green_report()
    r["damages"][0]["structural_suspicion"] = True
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert lane == "rojo"


def test_missing_estimacion_falls_to_amber(rules):
    r = _green_report()
    del r["estimacion"]  # placeholder state before Sprint 2
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("ambar", "AMBAR-1")


def test_valid_quality_no_damage_is_ambar2(rules):
    """Valid images but no damage detected → AMBAR-2 (possible false negative)."""
    r = _green_report()
    r["damages"] = []
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("ambar", "AMBAR-2")


def test_no_damage_invalid_quality_is_ambar1(rules):
    """No damage but invalid quality is the ordinary amber (AMBAR-1), not AMBAR-2."""
    r = _green_report()
    r["damages"] = []
    r["quality"] = {"valid": False}
    lane, rule_id, _ = triage.assign_lane(r, _meta(), rules)
    assert (lane, rule_id) == ("ambar", "AMBAR-1")


def test_lane_and_rule_id_contract(rules):
    lane, rule_id, reason = triage.assign_lane(_green_report(), _meta(), rules)
    assert lane in {"verde", "ambar", "rojo"}
    assert _RULE_ID_RE.match(rule_id)
    assert isinstance(reason, str) and reason
