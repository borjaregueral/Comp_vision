"""
Tests for scripts/quality_gate.py (T1.1).

Covers the happy path (valid image), the blocking edge cases required by the
plan (blurry, dark, low-resolution, no vehicle), EXIF stripping (RGPD), and the
error case (missing file). Vehicle detection is injected as a stub so the suite
stays fully offline (no YOLO download, no network).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import quality_gate as qg  # noqa: E402


# ── Vehicle-detector stubs (keep tests offline) ──────────────────────

def _vehicle_present(_image):
    # bbox spans the textured region of the synthetic fixtures so ROI-based
    # sharpness sees the checkerboard, not a flat corner.
    return {"vehicle_detected": True, "vehicle_area_fraction": 0.42,
            "n_vehicles": 1, "bbox": [40, 40, 760, 560]}


def _vehicle_absent(_image):
    return {"vehicle_detected": False, "vehicle_area_fraction": 0.0,
            "n_vehicles": 0, "bbox": None}


@pytest.fixture
def config():
    return qg.load_config()


# ── Individual metrics ───────────────────────────────────────────────

def test_sharp_image_above_threshold(synthetic_image, config):
    """Happy path: the textured synthetic image is clearly sharp."""
    variance = qg.assess_sharpness(synthetic_image)
    assert variance > config["sharpness"]["laplacian_variance_min"]


def test_blurry_image_below_threshold(blurry_image_file, config):
    image = cv2.imread(str(blurry_image_file))
    variance = qg.assess_sharpness(image)
    assert variance < config["sharpness"]["laplacian_variance_min"]


# ── assess_quality: blocking cases ───────────────────────────────────

def test_blurry_image_invalid(blurry_image_file, config):
    result = qg.assess_quality(
        blurry_image_file, config=config,
        vehicle_detector=_vehicle_present, strip_exif=False,
    )
    assert result["valid"] is False
    assert "blurry" in result["problems"]


def test_dark_image_invalid(dark_image_file, config):
    result = qg.assess_quality(
        dark_image_file, config=config,
        vehicle_detector=_vehicle_present, strip_exif=False,
    )
    assert result["valid"] is False
    assert "underexposed" in result["problems"]


def test_low_resolution_invalid(tmp_path, synthetic_image, config):
    small = cv2.resize(synthetic_image, (320, 240))
    path = tmp_path / "small.jpg"
    cv2.imwrite(str(path), small)
    result = qg.assess_quality(
        path, config=config,
        vehicle_detector=_vehicle_present, strip_exif=False,
    )
    assert result["valid"] is False
    assert "low_resolution" in result["problems"]


def test_no_vehicle_invalid(synthetic_image_file, config):
    result = qg.assess_quality(
        synthetic_image_file, config=config,
        vehicle_detector=_vehicle_absent, strip_exif=False,
    )
    assert result["valid"] is False
    assert "no_vehicle" in result["problems"]


def test_ok_image_valid(synthetic_image_file, config):
    """Happy path: sharp, well-exposed, big enough, vehicle present → valid."""
    result = qg.assess_quality(
        synthetic_image_file, config=config,
        vehicle_detector=_vehicle_present, strip_exif=False,
    )
    assert result["valid"] is True
    assert result["problems"] == []
    assert result["scores"]["sharpness"] > 0


# ── EXIF stripping (RGPD) ────────────────────────────────────────────

def _write_jpeg_with_exif(path: Path):
    from PIL import Image

    img = Image.new("RGB", (800, 600), (127, 127, 127))
    exif = img.getexif()
    exif[271] = "TestCameraMake"          # Make
    exif[272] = "TestCameraModel"         # Model
    exif[306] = "2020:01:01 10:00:00"     # DateTime
    img.save(path, exif=exif)


def test_extract_and_strip_exif_removes_identifying_fields(tmp_path):
    from PIL import Image

    path = tmp_path / "with_exif.jpg"
    _write_jpeg_with_exif(path)

    result = qg.extract_and_strip_exif(path)
    assert result["had_identifying_exif"] is True
    assert result["removed"] is True
    assert "Make" in result["identifying_fields"]
    assert "Model" in result["identifying_fields"]

    # Re-open: the identifying tags are gone.
    with Image.open(path) as cleaned:
        exif = cleaned.getexif()
    assert 271 not in exif
    assert 272 not in exif


def test_clean_image_reports_no_exif_removed(synthetic_image_file):
    result = qg.extract_and_strip_exif(synthetic_image_file)
    assert result["had_identifying_exif"] is False
    assert result["removed"] is False


def test_assess_quality_sets_exif_removed_flag(tmp_path, config):
    path = tmp_path / "with_exif.jpg"
    _write_jpeg_with_exif(path)
    result = qg.assess_quality(
        path, config=config, vehicle_detector=_vehicle_present, strip_exif=True,
    )
    assert result["exif_removed"] is True


# ── Vehicle detection parsing (fake model, no network) ───────────────

class _FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = np.array(cls, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.xyxy = np.array(xyxy, dtype=float)

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Minimal stand-in for an ultralytics YOLO model."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, **_kwargs):
        return [_FakeResult(self._boxes)]


def test_detect_vehicle_present_with_fake_model_detects_car():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    # One car (class 2) covering ~25% of the image (well above 10% threshold).
    boxes = _FakeBoxes(cls=[2], conf=[0.9], xyxy=[[100, 100, 500, 400]])
    out = qg.detect_vehicle_present(image, model=_FakeModel(boxes))
    assert out["vehicle_detected"] is True
    assert out["n_vehicles"] == 1
    assert out["vehicle_area_fraction"] == pytest.approx(0.25, abs=0.01)
    assert out["bbox"] == [100, 100, 500, 400]


def test_detect_vehicle_present_small_box_below_area_threshold():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    # Tiny car box (~1% of image) → below min_area_fraction → not "present".
    boxes = _FakeBoxes(cls=[2], conf=[0.9], xyxy=[[0, 0, 80, 60]])
    out = qg.detect_vehicle_present(image, model=_FakeModel(boxes))
    assert out["vehicle_detected"] is False


def test_detect_vehicle_present_overlapping_boxes_use_union_area():
    """#4: two overlapping boxes of the same car count once (union), not summed."""
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    # Two identical boxes (each ~25%). Summed would be ~0.50; union is ~0.25.
    boxes = _FakeBoxes(
        cls=[2, 2], conf=[0.9, 0.85],
        xyxy=[[100, 100, 500, 400], [100, 100, 500, 400]],
    )
    out = qg.detect_vehicle_present(image, model=_FakeModel(boxes))
    assert out["n_vehicles"] == 2
    assert out["vehicle_area_fraction"] == pytest.approx(0.25, abs=0.01)
    assert out["vehicle_area_fraction"] <= 1.0
    assert out["bbox"] == [100, 100, 500, 400]


# ── Sharpness ROI (#1) ───────────────────────────────────────────────

def test_assess_sharpness_roi_focuses_on_region(synthetic_image, config):
    """Sharpness over the textured region is high; over a flat corner is ~0."""
    min_var = config["sharpness"]["laplacian_variance_min"]
    textured = qg.assess_sharpness(synthetic_image, roi=[160, 120, 640, 480])
    flat = qg.assess_sharpness(synthetic_image, roi=[0, 0, 80, 80])
    assert textured > min_var
    assert flat < min_var


# ── Resolution orientation (#2) ──────────────────────────────────────

def test_portrait_image_not_rejected_for_resolution(tmp_path, synthetic_image, config):
    """#2: a 600x800 portrait carries the same detail as 800x600 landscape."""
    portrait = np.transpose(synthetic_image, (1, 0, 2)).copy()  # (800, 600, 3)
    assert portrait.shape[1] == 600 and portrait.shape[0] == 800
    path = tmp_path / "portrait.jpg"
    cv2.imwrite(str(path), portrait)
    result = qg.assess_quality(
        path, config=config, vehicle_detector=_vehicle_present, strip_exif=False,
    )
    assert "low_resolution" not in result["problems"]
    assert result["valid"] is True


# ── Error case ───────────────────────────────────────────────────────

def test_missing_file_raises(tmp_path, config):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError):
        qg.assess_quality(missing, config=config, vehicle_detector=_vehicle_present)
