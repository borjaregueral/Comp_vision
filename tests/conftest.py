"""
conftest.py — Shared fixtures for the fotoperitación test suite.

These provide deterministic, synthetic artifacts that Sprint 1+ tests consume
(quality gate, triage, output builder, orchestrator). No real claim data or PII
ever enters the test suite.

Fixtures:
- synthetic_image / synthetic_image_file: a valid synthetic image.
- blurry_image_file / dark_image_file: variants that must FAIL the quality gate.
- mock_prediction: damage-inference output with the exact contract returned by
  scripts/predict.run_inference (single frame).
- mock_claim_metadata: claim metadata exposing the fields consumed by the
  deterministic triage rules (business_rules/lane_rules.yaml).
"""

from pathlib import Path

import numpy as np
import pytest


# ── Synthetic images ─────────────────────────────────────────────────

@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Deterministic BGR image (H=600, W=800, 3) as uint8.

    Contains a high-contrast block so the Laplacian variance (sharpness) is
    clearly non-zero — i.e. this image is NOT blurry and should pass the
    sharpness check of the quality gate (T1.1).
    """
    h, w = 600, 800
    img = np.full((h, w, 3), 127, dtype=np.uint8)
    img[150:450, 200:600] = 30     # dark inner block
    img[200:400, 250:550] = 220    # bright core → strong edges
    return img


@pytest.fixture
def synthetic_image_file(tmp_path: Path, synthetic_image: np.ndarray) -> Path:
    """Write the synthetic image to a JPEG under tmp_path and return its path."""
    import cv2

    path = tmp_path / "synthetic_car.jpg"
    cv2.imwrite(str(path), synthetic_image)
    return path


@pytest.fixture
def blurry_image_file(tmp_path: Path, synthetic_image: np.ndarray) -> Path:
    """Heavily blurred variant → must FAIL the sharpness check (T1.1)."""
    import cv2

    blurred = cv2.GaussianBlur(synthetic_image, (31, 31), 0)
    path = tmp_path / "blurry_car.jpg"
    cv2.imwrite(str(path), blurred)
    return path


@pytest.fixture
def dark_image_file(tmp_path: Path) -> Path:
    """Near-black underexposed image → must FAIL the exposure check (T1.1)."""
    import cv2

    dark = np.full((600, 800, 3), 4, dtype=np.uint8)
    path = tmp_path / "dark_car.jpg"
    cv2.imwrite(str(path), dark)
    return path


# ── Model / pipeline outputs ─────────────────────────────────────────

@pytest.fixture
def mock_prediction() -> dict:
    """Single-frame damage-inference output.

    Mirrors the exact contract of scripts/predict.run_inference: a high-confidence
    'scratch' on a bumper. Used by triage / aggregation / output-builder tests.
    """
    return {
        "image": "synthetic_car.jpg",
        "image_path": "/tmp/synthetic_car.jpg",
        "image_size": [800, 600],
        "timestamp": "2026-06-06T12:00:00",
        "damages": [
            {
                "class": "scratch",
                "class_es": "Arañazo",
                "class_id": 1,
                "confidence": 0.91,
                "area_px": 5400,
                "area_pct": 1.13,
                "bbox": [210.0, 320.0, 540.0, 360.0],
                "mask_polygon": [[210, 320], [540, 320], [540, 360], [210, 360]],
            }
        ],
        "summary": {
            "total_damages": 1,
            "total_damage_area_pct": 1.13,
            "damage_types": ["scratch"],
            "severity": "Leve",
            "severity_en": "Minor",
        },
    }


@pytest.fixture
def mock_claim_metadata() -> dict:
    """Claim metadata with the fields consumed by the triage rules
    (business_rules/lane_rules.yaml). No PII (no plate, no name)."""
    return {
        "claim_id": "TEST-CLAIM-0001",
        "marca": "Seat",
        "modelo": "Ibiza",
        "anio": 2019,
        "color": "blanco",
        "provincia": "Zaragoza",
        "valor_vehiculo_estimado": 12000,
        "siniestros_12m": 1,
        "descripcion_asegurado": "Rayón en el paragolpes delantero al aparcar.",
    }
