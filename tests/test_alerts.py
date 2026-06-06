"""
Tests for scripts/alerts.py (T2.4).

Covers the plan case (declared "paragolpes" but damage on a door -> mismatch),
the other heuristics, the disabled manipulation placeholder, alert-shape
conformance with the output schema, and the integration guarantee that a
critical alert forces the red lane via triage (ROJO-3).
"""

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import alerts  # noqa: E402
import triage  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "inference_output_v1.json"


def _door_damage():
    return {"part": "front_left_door", "part_category": "body_panel",
            "zone": "front_left", "type": "scratch"}


# ── Part-declaration mismatch (plan case) ─────────────────────────────

def test_mismatch_bumper_declared_door_detected():
    meta = {"descripcion_asegurado": "Golpe en el paragolpes delantero al aparcar."}
    out = alerts.detect_alerts([_door_damage()], meta)
    ids = {a["id"] for a in out}
    assert "part_declaration_mismatch" in ids


def test_no_mismatch_when_declared_part_matches():
    meta = {"descripcion_asegurado": "Rayón en el paragolpes delantero."}
    bumper = {"part": "front_bumper", "part_category": "plastic_panel",
              "zone": "front", "type": "scratch"}
    out = alerts.detect_alerts([bumper], meta)
    assert "part_declaration_mismatch" not in {a["id"] for a in out}


def test_no_description_no_mismatch():
    out = alerts.detect_alerts([_door_damage()], {"descripcion_asegurado": ""})
    assert "part_declaration_mismatch" not in {a["id"] for a in out}


def test_mismatch_is_accent_insensitive():
    # 'capó' in description, accent-stripped, vs a door damage → mismatch.
    meta = {"descripcion_asegurado": "Daño en el capó"}
    assert alerts.alert_part_declaration_mismatch([_door_damage()], meta["descripcion_asegurado"]) is not None


# ── Multiple unrelated damages ────────────────────────────────────────

def test_multiple_unrelated_triggers():
    damages = [
        {"zone": "front_left", "type": "scratch"},
        {"zone": "rear_right", "type": "dent"},
        {"zone": "front", "type": "crack"},
    ]
    out = alerts.alert_multiple_unrelated_damages(damages)
    assert out is not None and out["id"] == "multiple_unrelated_damages"


def test_multiple_unrelated_not_triggered_for_few_zones():
    damages = [{"zone": "front", "type": "scratch"}, {"zone": "front", "type": "dent"}]
    assert alerts.alert_multiple_unrelated_damages(damages) is None


# ── Preexisting damage heuristic ──────────────────────────────────────

def test_preexisting_flags_rusty_crop():
    rusty = np.zeros((20, 20, 3), dtype=np.uint8)
    rusty[:] = (20, 70, 150)  # BGR orange/brown → rust HSV range
    out = alerts.alert_preexisting_damage(rusty)
    assert out is not None and out["id"] == "preexisting_damage_suspected"


def test_preexisting_ignores_clean_grey_crop():
    grey = np.full((20, 20, 3), 127, dtype=np.uint8)
    assert alerts.alert_preexisting_damage(grey) is None


def test_preexisting_none_without_crop():
    assert alerts.alert_preexisting_damage(None) is None


# ── Image manipulation placeholder ────────────────────────────────────

def test_image_manipulation_placeholder_disabled():
    assert alerts.alert_image_manipulation(["a.jpg"]) is None


# ── Alert shape conforms to the output schema ─────────────────────────

def test_alerts_conform_to_schema():
    from jsonschema import Draft202012Validator

    meta = {"descripcion_asegurado": "Golpe en el paragolpes"}
    out = alerts.detect_alerts([_door_damage()], meta)
    assert out  # at least the mismatch alert
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        alerts_schema = json.load(fh)["properties"]["alerts"]
    Draft202012Validator(alerts_schema).validate(out)  # raises if non-conformant
    for a in out:
        assert a["severity"] in {"info", "warning", "critical"}


# ── Critical alert forces the red lane (integration with triage) ──────

def test_critical_alert_forces_red_lane():
    rules = triage.load_rules()
    report = {
        "quality": {"valid": True},
        "damages": [{"confidence": 0.9, "structural_suspicion": False}],
        "estimacion": {"total_eur": 200.0, "confidence_overall": 0.9},
        "alerts": [{"id": "fraud_signal", "severity": "critical", "description": "x"}],
    }
    lane, rule_id, _ = triage.assign_lane(report, {"siniestros_12m": 1}, rules)
    assert lane == "rojo" and rule_id == "ROJO-3"


def test_v1_heuristics_do_not_emit_critical():
    """v1 heuristics stay at 'warning' so they never auto-force a red lane."""
    meta = {"descripcion_asegurado": "Golpe en el paragolpes"}
    out = alerts.detect_alerts([_door_damage()], meta)
    assert all(a["severity"] != "critical" for a in out)
