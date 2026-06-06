"""
Tests for scripts/output_builder.py (T1.2).

Covers the happy path (a fully-formed output validates), the id/timestamp
generation contract, and the error cases required by the plan: a missing
required field and an out-of-enum value must raise an explicit, readable
OutputValidationError (never a silent pass).
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import output_builder as ob  # noqa: E402

_HASH = "a" * 64  # stand-in SHA256 hex


def _valid_kwargs() -> dict:
    """Fully-formed, schema-valid parts for build_output."""
    return dict(
        claim_id="SIN-2026-000123",
        model_version={"damage_model": "baseline_v1.0", "parts_model": "parts_seg_v1.0"},
        quality={
            "valid": True,
            "per_image": [{"image_hash": _HASH, "valid": True, "problems": []}],
        },
        damages=[{
            "damage_id": "D1",
            "type": "scratch",
            "zone": "front_left",
            "part": "front_left_door",
            "extension": "small",
            "severity": "leve",
            "confidence": 0.91,
            "supporting_images": [_HASH],
        }],
        alerts=[],
        estimacion={
            "total_eur": 320.0,
            "p25_eur": 280.0,
            "p75_eur": 380.0,
            "breakdown": {"mano_obra": 200.0, "piezas": 0.0, "materiales": 50.0, "iva": 70.0},
            "confidence_overall": 0.8,
            "currency": "EUR",
            "iva_included": True,
        },
        lane="verde",
        lane_rule_id="VERDE-1",
        lane_reason="Caso simple sin alertas, importe bajo.",
        next_action="Liquidar automáticamente y notificar al asegurado.",
        audit={"input_hashes": [_HASH], "processing_time_ms": 1234},
    )


# ── Happy path ───────────────────────────────────────────────────────

def test_build_output_valid_passes():
    out = ob.build_output(**_valid_kwargs())
    assert out["schema_version"] == "1.0.0"
    assert out["lane"] == "verde"
    assert re.match(r"^EVA-[0-9]{8}-[A-F0-9]{8}$", out["id_evaluacion"])
    assert out["timestamp"].endswith("Z")
    # No optional zones_summary was passed → it must be absent (additionalProperties).
    assert "zones_summary" not in out


def test_build_output_with_zones_summary_included():
    out = ob.build_output(**_valid_kwargs(), zones_summary={"front_left": 1})
    assert out["zones_summary"] == {"front_left": 1}


def test_generate_evaluation_id_matches_pattern():
    eval_id = ob.generate_evaluation_id("2026-06-06T10:00:00Z")
    assert eval_id.startswith("EVA-20260606-")
    assert re.match(r"^EVA-[0-9]{8}-[A-F0-9]{8}$", eval_id)


def test_explicit_id_and_timestamp_preserved():
    out = ob.build_output(
        **_valid_kwargs(),
        id_evaluacion="EVA-20260606-ABCDEF12",
        timestamp="2026-06-06T09:30:00Z",
    )
    assert out["id_evaluacion"] == "EVA-20260606-ABCDEF12"
    assert out["timestamp"] == "2026-06-06T09:30:00Z"


# ── Error cases (explicit, readable failures) ────────────────────────

def test_missing_required_field_raises_with_clear_message():
    kwargs = _valid_kwargs()
    del kwargs["estimacion"]["total_eur"]  # drop a required nested field
    with pytest.raises(ob.OutputValidationError) as exc:
        ob.build_output(**kwargs)
    msg = str(exc.value)
    assert "total_eur" in msg
    assert "schema validation" in msg


def test_missing_top_level_field_raises():
    out = ob.build_output(**_valid_kwargs())
    del out["next_action"]
    with pytest.raises(ob.OutputValidationError) as exc:
        ob.validate_output(out)
    assert "next_action" in str(exc.value)


def test_bad_lane_enum_raises():
    kwargs = _valid_kwargs()
    kwargs["lane"] = "azul"  # not in {verde, ambar, rojo}
    with pytest.raises(ob.OutputValidationError) as exc:
        ob.build_output(**kwargs)
    assert "lane" in str(exc.value)


def test_bad_lane_rule_id_pattern_raises():
    kwargs = _valid_kwargs()
    kwargs["lane_rule_id"] = "GREEN-1"  # must match ^(VERDE|AMBAR|ROJO)-[0-9]+$
    with pytest.raises(ob.OutputValidationError):
        ob.build_output(**kwargs)


def test_additional_property_rejected():
    out = ob.build_output(**_valid_kwargs())
    out["unexpected_field"] = "x"  # additionalProperties: false at root
    with pytest.raises(ob.OutputValidationError) as exc:
        ob.validate_output(out)
    assert "unexpected_field" in str(exc.value) or "additional" in str(exc.value).lower()


def test_validate_can_be_skipped():
    """build_output(validate=False) returns even an incomplete object."""
    kwargs = _valid_kwargs()
    kwargs["lane"] = "azul"
    out = ob.build_output(**kwargs, validate=False)
    assert out["lane"] == "azul"  # not validated, returned as-is
