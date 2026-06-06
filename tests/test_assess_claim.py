"""
Tests for scripts/assess_claim.py — the integrated pipeline (T1.5 + T2.6).

Runs the orchestrator end-to-end on synthetic images with an injected (offline)
damage detector. After T2.6 it exercises the real aggregation, cost estimate,
economic severity, alerts and triage. Includes the Sprint 2 acceptance case.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import assess_claim as ac  # noqa: E402
import audit_log  # noqa: E402
import output_builder  # noqa: E402


def _sharp_image() -> np.ndarray:
    img = np.full((600, 800, 3), 127, dtype=np.uint8)
    tile = 8
    for y in range(120, 480, tile):
        for x in range(160, 640, tile):
            img[y:y + tile, x:x + tile] = 70 if ((x // tile) + (y // tile)) % 2 == 0 else 185
    return img


def _make_images(tmp_path: Path, n: int) -> list:
    paths = []
    for i in range(n):
        img = _sharp_image()
        img[0:8, 0:8] = (i * 50) % 255  # make each photo unique → distinct hashes
        p = tmp_path / f"img_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _vehicle_present(_image):
    return {"vehicle_detected": True, "vehicle_area_fraction": 0.42,
            "n_vehicles": 1, "bbox": [40, 40, 760, 560]}


def _vehicle_absent(_image):
    return {"vehicle_detected": False, "vehicle_area_fraction": 0.0, "n_vehicles": 0, "bbox": None}


def _scratch_detector(_image_path):
    return [{"type": "scratch", "part": "front_bumper", "part_category": "plastic_panel",
             "zone": "front", "extension": "medium", "confidence": 0.90, "area_pct": 1.5}]


def _replace_detector(_image_path):
    return [{"type": "dent", "part": "front_bumper", "part_category": "plastic_panel",
             "zone": "front", "extension": "large", "confidence": 0.90, "area_pct": 8.0}]


def _structural_detector(_image_path):
    return [{"type": "crack", "part": "front_left_door", "part_category": "body_panel",
             "zone": "front_left", "extension": "medium", "confidence": 0.90, "area_pct": 3.0}]


def _no_damage_detector(_image_path):
    return []


_META = {"valor_vehiculo_estimado": 12000, "siniestros_12m": 1}
_MODEL_VERSION = {"damage_model": "test_model_v0", "parts_model": "test_parts_v0"}


def _run(tmp_path, images, *, detector=_scratch_detector, metadata=None, **over):
    kwargs = dict(
        damage_detector=detector,
        quality_vehicle_detector=_vehicle_present,
        model_version=_MODEL_VERSION,
        log_dir=tmp_path / "logs",
    )
    kwargs.update(over)
    return ac.assess_claim("SIN-TEST-1", images, metadata or _META, **kwargs)


# ── Sprint 2 acceptance ───────────────────────────────────────────────

def test_sprint2_acceptance_four_photos(tmp_path):
    images = _make_images(tmp_path, 4)
    output = _run(tmp_path, images)
    output_builder.validate_output(output)

    est = output["estimacion"]
    assert est["total_eur"] > 0
    assert est["p25_eur"] <= est["total_eur"] <= est["p75_eur"]
    assert set(est["breakdown"]) == {"mano_obra", "piezas", "materiales", "iva"}

    assert len(output["damages"]) == 1            # 4 views of one damage → 1
    assert output["damages"][0]["severity"] in {"leve", "moderado", "severo"}
    assert isinstance(output["alerts"], list)
    assert output["lane"] in {"verde", "ambar", "rojo"}

    # Audit line written with the table versions used.
    log_files = list((tmp_path / "logs").glob("inference_*.jsonl"))
    assert len(log_files) == 1
    record = json.loads(log_files[0].read_text().splitlines()[0])
    assert record["output_summary"]["total_eur"] == est["total_eur"]
    assert record["output_summary"]["n_damages"] == 1


def test_three_photos_aggregate_to_one(tmp_path):
    images = _make_images(tmp_path, 3)
    output = _run(tmp_path, images)
    assert len(output["damages"]) == 1
    assert len(output["damages"][0]["supporting_images"]) == 3
    assert len(output["audit"]["input_hashes"]) == 3


# ── Lane behaviour with real cost / alerts ────────────────────────────

def test_clean_simple_case_is_green(tmp_path):
    """Single cheap scratch, high confidence, no alerts → auto-resolve (verde)."""
    output = _run(tmp_path, _make_images(tmp_path, 2))
    assert output["lane"] == "verde"


def test_mismatch_alert_blocks_green(tmp_path):
    meta = {**_META, "descripcion_asegurado": "Daño en el faro delantero"}
    output = _run(tmp_path, _make_images(tmp_path, 2), metadata=meta)
    ids = {a["id"] for a in output["alerts"]}
    assert "part_declaration_mismatch" in ids
    assert output["lane"] == "ambar"  # a warning alert blocks the green lane


def test_replace_part_is_priced(tmp_path):
    meta = {**_META, "marca": "Seat", "modelo": "Ibiza"}
    output = _run(tmp_path, _make_images(tmp_path, 1), detector=_replace_detector, metadata=meta)
    assert output["estimacion"]["breakdown"]["piezas"] > 0


def test_structural_damage_forces_red(tmp_path):
    output = _run(tmp_path, _make_images(tmp_path, 1), detector=_structural_detector)
    assert output["lane"] == "rojo"
    assert output["lane_rule_id"] == "ROJO-1"


def test_valid_image_no_damage_is_amber2(tmp_path):
    """User-approved rule: valid images with no detected damage → AMBAR-2."""
    output = _run(tmp_path, _make_images(tmp_path, 2), detector=_no_damage_detector)
    output_builder.validate_output(output)
    assert output["damages"] == []
    assert output["lane"] == "ambar"
    assert output["lane_rule_id"] == "AMBAR-2"


def test_no_valid_images_is_amber(tmp_path):
    output = _run(tmp_path, _make_images(tmp_path, 2), quality_vehicle_detector=_vehicle_absent)
    output_builder.validate_output(output)
    assert output["quality"]["valid"] is False
    assert output["damages"] == []
    assert output["lane"] == "ambar"
    assert len(output["audit"]["input_hashes"]) == 2


# ── Robustness ────────────────────────────────────────────────────────

def test_empty_images_raises(tmp_path):
    with pytest.raises(ValueError):
        ac.assess_claim("SIN-X", [], _META, damage_detector=_scratch_detector,
                        quality_vehicle_detector=_vehicle_present, log_dir=tmp_path)


def test_input_hashes_match_files(tmp_path):
    images = _make_images(tmp_path, 3)
    output = _run(tmp_path, images)
    assert output["audit"]["input_hashes"] == [audit_log.hash_image(p) for p in images]


def test_default_damage_detector_raises_when_weights_absent():
    with pytest.raises(FileNotFoundError):
        ac._default_damage_detector(model_path="models/does_not_exist/best.pt")


def test_find_images_discovers_files(tmp_path):
    images = _make_images(tmp_path, 2)
    assert {p.name for p in ac._find_images(tmp_path)} == {p.name for p in images}
