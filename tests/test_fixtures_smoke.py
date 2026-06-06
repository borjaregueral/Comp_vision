"""
Smoke tests for the conftest fixtures.

These verify that the synthetic artifacts used across the suite are generated
correctly. They do NOT test business logic yet — that arrives with the Sprint 1
modules. Covers the happy path plus edge/contract checks per the testing rule.
"""

from pathlib import Path

import numpy as np


def test_synthetic_image_shape(synthetic_image):
    """Happy path: synthetic image has the expected shape and dtype."""
    assert synthetic_image.shape == (600, 800, 3)
    assert synthetic_image.dtype == np.uint8


def test_synthetic_image_file_written(synthetic_image_file):
    """The valid image is persisted as a non-empty file on disk."""
    assert isinstance(synthetic_image_file, Path)
    assert synthetic_image_file.exists()
    assert synthetic_image_file.stat().st_size > 0


def test_quality_gate_negative_fixtures_exist(blurry_image_file, dark_image_file):
    """Edge cases: the must-fail fixtures (blurry, dark) are produced on disk."""
    assert blurry_image_file.exists()
    assert dark_image_file.exists()


def test_mock_prediction_contract(mock_prediction):
    """mock_prediction matches the predict.run_inference contract."""
    assert mock_prediction["summary"]["total_damages"] == len(mock_prediction["damages"])
    damage = mock_prediction["damages"][0]
    for key in ("class", "class_id", "confidence", "bbox", "mask_polygon"):
        assert key in damage
    assert 0.0 <= damage["confidence"] <= 1.0


def test_mock_claim_metadata_has_triage_fields(mock_claim_metadata):
    """Metadata exposes the fields the triage rules read, and carries no PII."""
    for key in ("claim_id", "valor_vehiculo_estimado", "siniestros_12m", "descripcion_asegurado"):
        assert key in mock_claim_metadata
    # Guard against accidentally seeding PII into the test suite.
    assert "matricula" not in mock_claim_metadata
    assert "nombre" not in mock_claim_metadata
