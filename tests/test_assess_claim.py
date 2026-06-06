"""
Tests for scripts/assess_claim.py (T1.5) — the Sprint 1 acceptance test.

Runs the orchestrator end-to-end on synthetic images with injected (offline)
model components: it must produce a schema-valid output and leave exactly one
audit line. Also covers the no-valid-image path, empty input, real triage
integration (structural → red) and input-hash traceability.
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
    img = _sharp_image()
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _vehicle_present(_image):
    return {"vehicle_detected": True, "vehicle_area_fraction": 0.42,
            "n_vehicles": 1, "bbox": [40, 40, 760, 560]}


def _vehicle_absent(_image):
    return {"vehicle_detected": False, "vehicle_area_fraction": 0.0, "n_vehicles": 0, "bbox": None}


def _fake_detector(_image_path):
    return [{"type": "scratch", "confidence": 0.90, "area_pct": 1.2}]


_META = {"valor_vehiculo_estimado": 12000, "siniestros_12m": 1}
_MODEL_VERSION = {"damage_model": "test_model_v0", "parts_model": "test_parts_v0"}


def _run(tmp_path, images, **over):
    kwargs = dict(
        damage_detector=_fake_detector,
        quality_vehicle_detector=_vehicle_present,
        model_version=_MODEL_VERSION,
        log_dir=tmp_path / "logs",
    )
    kwargs.update(over)
    return ac.assess_claim("SIN-TEST-1", images, _META, **kwargs)


# ── Sprint 1 acceptance: end-to-end valid output + audit line ─────────

def test_end_to_end_produces_valid_output(tmp_path):
    images = _make_images(tmp_path, 3)
    output = _run(tmp_path, images)
    # build_output validated internally; assert it again explicitly (no raise).
    output_builder.validate_output(output)
    assert output["claim_id"] == "SIN-TEST-1"
    assert output["lane"] in {"verde", "ambar", "rojo"}
    assert len(output["damages"]) == 3            # one per valid image (placeholder aggregation)
    assert len(output["audit"]["input_hashes"]) == 3


def test_end_to_end_writes_one_audit_line(tmp_path):
    images = _make_images(tmp_path, 3)
    output = _run(tmp_path, images)
    log_files = list((tmp_path / "logs").glob("inference_*.jsonl"))
    assert len(log_files) == 1
    lines = log_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["output_summary"]["n_damages"] == 3
    assert record["output_summary"]["lane"] == output["lane"]
    assert record["id_evaluacion"] == output["id_evaluacion"]


def test_placeholder_estimate_keeps_claim_out_of_green(tmp_path):
    """With the zeroed placeholder estimate (confidence 0), no auto-green."""
    images = _make_images(tmp_path, 2)
    output = _run(tmp_path, images)
    assert output["lane"] != "verde"


# ── Robustness / edges ────────────────────────────────────────────────

def test_no_valid_images_is_amber_and_valid(tmp_path):
    images = _make_images(tmp_path, 2)
    output = _run(tmp_path, images, quality_vehicle_detector=_vehicle_absent)
    output_builder.validate_output(output)
    assert output["quality"]["valid"] is False
    assert output["damages"] == []
    assert output["lane"] == "ambar"
    # All input images are still hashed for traceability.
    assert len(output["audit"]["input_hashes"]) == 2


def test_empty_images_raises(tmp_path):
    with pytest.raises(ValueError):
        ac.assess_claim("SIN-X", [], _META, damage_detector=_fake_detector,
                        quality_vehicle_detector=_vehicle_present, log_dir=tmp_path)


def test_structural_damage_forces_red(tmp_path):
    images = _make_images(tmp_path, 1)

    def structural_detector(_image_path):
        return [{"type": "crack", "confidence": 0.9, "area_pct": 4.0, "structural_suspicion": True}]

    output = _run(tmp_path, images, damage_detector=structural_detector)
    assert output["lane"] == "rojo"
    assert output["lane_rule_id"] == "ROJO-1"


def test_input_hashes_match_files(tmp_path):
    images = _make_images(tmp_path, 3)
    output = _run(tmp_path, images)
    expected = [audit_log.hash_image(p) for p in images]
    assert output["audit"]["input_hashes"] == expected


def test_default_damage_detector_raises_when_weights_absent():
    """Production path: clear error while best.pt is still on the remote GPU (T0.1)."""
    with pytest.raises(FileNotFoundError):
        ac._default_damage_detector(model_path="models/does_not_exist/best.pt")


def test_find_images_discovers_files(tmp_path):
    images = _make_images(tmp_path, 2)
    found = ac._find_images(tmp_path)
    assert {p.name for p in found} == {p.name for p in images}
