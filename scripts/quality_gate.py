#!/usr/bin/env python3
"""
quality_gate.py — Image quality filter applied BEFORE damage inference.

Rejects images that are unusable for automatic appraisal — too blurry, badly
exposed, too small, or without a vehicle — so the damage detector never emits
high-confidence predictions on garbage inputs. It also strips identifying EXIF
metadata (RGPD / privacy, rule 13: anonymization by default).

This module is a pure *input* stage: it does NOT decide the triage lane (that is
scripts/triage.py, deterministic, downstream). It only produces an objective
quality verdict that the orchestrator consumes.

All thresholds live in configs/quality_gate.yaml (config out of code, rule 7).

Public API
----------
    load_config(path=None) -> dict
    assess_sharpness(image) -> float
    assess_exposure(image) -> dict
    assess_resolution(image) -> dict
    detect_vehicle_present(image, model=None, ...) -> dict
    extract_and_strip_exif(image_path, output_path=None) -> dict
    assess_quality(image_path, config=None, vehicle_detector=None,
                   strip_exif=True, check_vehicle=True)
        -> {"valid": bool, "problems": list[str], "scores": dict,
            "exif_removed": bool}

CLI
---
    python scripts/quality_gate.py --source image_or_dir/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger("quality_gate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "quality_gate.yaml"

# EXIF tags considered personally identifying (RGPD). Numbers are standard EXIF
# tag IDs as exposed by Pillow's ExifTags. Kept explicit so an auditor can see
# exactly what we strip.
_IDENTIFYING_EXIF_TAGS = {
    271: "Make",
    272: "Model",
    305: "Software",
    306: "DateTime",
    315: "Artist",
    316: "HostComputer",
    33432: "Copyright",
    37510: "UserComment",
    42016: "ImageUniqueID",
    34853: "GPSInfo",
}
# Tags living in the Exif sub-IFD (0x8769).
_IDENTIFYING_EXIF_SUBIFD = {
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    42032: "CameraOwnerName",
    42033: "BodySerialNumber",
    42034: "LensSpecification",
}
_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769


# ── Config ───────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> dict:
    """Load the quality-gate thresholds from YAML.

    Args:
        path: Optional override path. Defaults to configs/quality_gate.yaml.

    Returns:
        Parsed config dict.
    """
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Quality-gate config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Image loading ────────────────────────────────────────────────────

def _read_image(image_path: Path) -> np.ndarray:
    """Read an image as a BGR ndarray, raising on missing/corrupt files.

    Mirrors the integrity intent of download_datasets.is_valid_image: a file
    that OpenCV cannot decode is treated as unusable rather than silently empty.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image could not be decoded (corrupt/unsupported): {image_path}")
    return image


# ── Individual quality metrics ───────────────────────────────────────

def assess_sharpness(image: np.ndarray, roi=None) -> float:
    """Return the variance of the Laplacian — the classic blur metric.

    A sharp image has strong high-frequency content → high variance. A blurry
    one is smooth → low variance.

    Args:
        image: BGR image.
        roi: Optional [x1, y1, x2, y2] region (e.g. the vehicle bounding box).
            When given, sharpness is measured ONLY inside it, so a large smooth
            background (asphalt, sky) does not dilute the score. Falls back to
            the whole image if the ROI is empty/degenerate.

    Note:
        This metric still under-reports on extreme close-ups of smooth painted
        panels (little texture even when in focus). The threshold is therefore
        deliberately low; real-photo calibration is pending (see model card).
    """
    target = image
    if roi is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            target = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_exposure(image: np.ndarray) -> dict:
    """Measure exposure: fraction of near-clipped pixels and mean brightness.

    Returns:
        {saturated_high_pct, saturated_low_pct, mean_brightness} — percentages
        in [0, 100] and brightness in [0, 255].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total = gray.size
    high_pct = float(np.count_nonzero(gray > 250) / total * 100.0)
    low_pct = float(np.count_nonzero(gray < 15) / total * 100.0)
    return {
        "saturated_high_pct": round(high_pct, 3),
        "saturated_low_pct": round(low_pct, 3),
        "mean_brightness": round(float(gray.mean()), 2),
    }


def assess_resolution(image: np.ndarray) -> dict:
    """Return image dimensions and megapixels."""
    h, w = image.shape[:2]
    return {"width": int(w), "height": int(h), "megapixels": round(w * h / 1e6, 2)}


def detect_vehicle_present(
    image: np.ndarray,
    model=None,
    model_path: str = "yolo11n.pt",
    target_classes=(2, 5, 7),
    min_confidence: float = 0.50,
    min_area_fraction: float = 0.10,
) -> dict:
    """Detect whether a vehicle occupies a meaningful fraction of the image.

    Uses a COCO-pretrained YOLO (yolo11n.pt by default). ``model`` can be
    injected (any object exposing the ultralytics ``predict`` interface) to keep
    this offline-testable; when None, the model is lazily loaded.

    Returns:
        {vehicle_detected, vehicle_area_fraction, n_vehicles}
    """
    if model is None:
        from ultralytics import YOLO  # lazy: avoids import/download at module load
        model = YOLO(model_path)

    img_h, img_w = image.shape[:2]
    results = model.predict(
        source=image,
        conf=min_confidence,
        classes=list(target_classes),
        verbose=False,
    )

    n_vehicles = 0
    # Rasterize qualifying boxes into a mask so overlapping detections of the
    # same car are counted once (union area), not summed — a sum can exceed 1.0.
    mask = np.zeros((img_h, img_w), dtype=bool)
    ux1 = uy1 = float("inf")
    ux2 = uy2 = float("-inf")
    if results:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                if cls_id not in target_classes or conf < min_confidence:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
                xi2, yi2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                mask[yi1:yi2, xi1:xi2] = True
                ux1, uy1 = min(ux1, xi1), min(uy1, yi1)
                ux2, uy2 = max(ux2, xi2), max(uy2, yi2)
                n_vehicles += 1

    union_area = int(mask.sum())
    area_fraction = union_area / float(img_w * img_h) if img_w and img_h else 0.0
    detected = n_vehicles > 0 and area_fraction >= min_area_fraction
    bbox = [int(ux1), int(uy1), int(ux2), int(uy2)] if n_vehicles > 0 else None
    return {
        "vehicle_detected": bool(detected),
        "vehicle_area_fraction": round(float(area_fraction), 4),
        "n_vehicles": int(n_vehicles),
        "bbox": bbox,
    }


# ── EXIF stripping (RGPD) ────────────────────────────────────────────

def _identifying_exif_fields(exif) -> set:
    """Return the set of identifying field names present in an Exif object."""
    found = set()
    for tag, name in _IDENTIFYING_EXIF_TAGS.items():
        if tag in exif:
            found.add(name)
    try:
        if exif.get_ifd(_GPS_IFD):
            found.add("GPSInfo")
    except Exception:  # pragma: no cover - defensive against odd EXIF blobs
        pass
    try:
        sub = exif.get_ifd(_EXIF_IFD)
        for tag, name in _IDENTIFYING_EXIF_SUBIFD.items():
            if tag in sub:
                found.add(name)
    except Exception:  # pragma: no cover
        pass
    return found


def extract_and_strip_exif(image_path, output_path=None) -> dict:
    """Detect identifying EXIF metadata and write a metadata-free copy.

    By default (output_path=None) the image is stripped **in place** — this is
    the anonymization-by-default policy (rule 13). Pass output_path to write the
    cleaned image elsewhere and leave the original untouched.

    Returns:
        {had_identifying_exif, identifying_fields, removed, output_path}
    """
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img.load()
        # Copy pixels into an independent array; rebuilding from it drops every
        # metadata block (EXIF, GPS, maker notes) without carrying the original
        # file's info dict.
        pixels = np.array(img)
        identifying = _identifying_exif_fields(img.getexif())

    had = len(identifying) > 0
    target = Path(output_path) if output_path else image_path
    removed = False

    # Only rewrite when there is something to strip, or when an explicit output
    # copy was requested. Avoids needless recompression of already-clean files.
    if had or output_path:
        clean = Image.fromarray(pixels)
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            clean.save(target, quality=95)
        else:
            clean.save(target)
        removed = had

    return {
        "had_identifying_exif": had,
        "identifying_fields": sorted(identifying),
        "removed": removed,
        "output_path": str(target) if (had or output_path) else None,
    }


# ── Orchestration ────────────────────────────────────────────────────

def assess_quality(
    image_path,
    config: Optional[dict] = None,
    vehicle_detector: Optional[Callable[[np.ndarray], dict]] = None,
    strip_exif: bool = True,
    check_vehicle: bool = True,
) -> dict:
    """Run the full quality gate over a single image.

    Args:
        image_path: Path to the image.
        config: Parsed quality_gate config; loaded from YAML if None.
        vehicle_detector: Optional callable (image)->detect_vehicle_present-like
            dict. Injected in tests to stay offline; defaults to a real YOLO.
        strip_exif: If True (and config enables it), strip identifying EXIF
            in place before returning. Sets ``exif_removed`` accordingly.
        check_vehicle: If False, skip the vehicle-presence check entirely.

    Returns:
        {"valid": bool, "problems": list[str], "scores": dict,
         "exif_removed": bool}

    Raises:
        FileNotFoundError: if image_path does not exist.
        ValueError: if the image cannot be decoded.
    """
    config = config or load_config()
    image_path = Path(image_path)
    image = _read_image(image_path)

    problems: list[str] = []
    scores: dict = {}

    # ── Vehicle presence ──
    # Run first: its bounding box focuses the sharpness check below on the
    # vehicle instead of the (often smooth) background.
    vehicle = None
    veh_cfg = config.get("vehicle_present", {})
    if check_vehicle and veh_cfg.get("enabled", True):
        if vehicle_detector is None:
            def _default_vehicle_detector(img):  # local default, real YOLO path
                return detect_vehicle_present(
                    img,
                    model_path=veh_cfg.get("model", "yolo11n.pt"),
                    target_classes=tuple(veh_cfg.get("target_classes", (2, 5, 7))),
                    min_confidence=veh_cfg.get("min_confidence", 0.50),
                    min_area_fraction=veh_cfg.get("min_area_fraction", 0.10),
                )
            vehicle_detector = _default_vehicle_detector
        vehicle = vehicle_detector(image)
        scores["vehicle_detected"] = bool(vehicle.get("vehicle_detected", False))
        scores["vehicle_area_fraction"] = vehicle.get("vehicle_area_fraction", 0.0)
        if veh_cfg.get("blocking", True) and not vehicle.get("vehicle_detected", False):
            problems.append("no_vehicle")

    # ── Sharpness (measured on the vehicle ROI when available) ──
    sharp_cfg = config.get("sharpness", {})
    if sharp_cfg.get("enabled", True):
        roi = vehicle.get("bbox") if (vehicle and sharp_cfg.get("use_vehicle_roi", True)) else None
        variance = assess_sharpness(image, roi=roi)
        scores["sharpness"] = round(variance, 2)
        scores["sharpness_scope"] = "vehicle_roi" if roi else "full_image"
        if sharp_cfg.get("blocking", True) and variance < sharp_cfg.get("laplacian_variance_min", 80.0):
            problems.append("blurry")

    # ── Exposure ──
    exp_cfg = config.get("exposure", {})
    if exp_cfg.get("enabled", True):
        exposure = assess_exposure(image)
        scores.update(exposure)
        if exp_cfg.get("blocking", True):
            if exposure["saturated_high_pct"] > exp_cfg.get("saturated_high_pct_max", 15.0):
                problems.append("overexposed")
            if exposure["saturated_low_pct"] > exp_cfg.get("saturated_low_pct_max", 25.0):
                problems.append("underexposed")
            mean = exposure["mean_brightness"]
            if mean < exp_cfg.get("mean_brightness_min", 40) and "underexposed" not in problems:
                problems.append("underexposed")
            if mean > exp_cfg.get("mean_brightness_max", 220) and "overexposed" not in problems:
                problems.append("overexposed")

    # ── Resolution (orientation-independent) ──
    # Adequacy must not depend on portrait vs landscape: an 800x600 photo and
    # its 600x800 rotation carry the same detail, so compare the long/short
    # sides against the long/short minimums.
    res_cfg = config.get("resolution", {})
    if res_cfg.get("enabled", True):
        resolution = assess_resolution(image)
        scores.update(resolution)
        if res_cfg.get("blocking", True):
            long_min = max(res_cfg.get("min_width", 800), res_cfg.get("min_height", 600))
            short_min = min(res_cfg.get("min_width", 800), res_cfg.get("min_height", 600))
            long_side = max(resolution["width"], resolution["height"])
            short_side = min(resolution["width"], resolution["height"])
            if long_side < long_min or short_side < short_min:
                problems.append("low_resolution")

    # ── EXIF stripping (RGPD) ──
    exif_removed = False
    exif_cfg = config.get("exif", {})
    if strip_exif and exif_cfg.get("strip_on_ingest", True):
        exif_result = extract_and_strip_exif(image_path)
        exif_removed = exif_result["removed"]
        if exif_result["had_identifying_exif"]:
            scores["exif_identifying_fields"] = exif_result["identifying_fields"]

    valid = len(problems) == 0
    return {
        "valid": valid,
        "problems": problems,
        "scores": scores,
        "exif_removed": exif_removed,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def _find_images(source: Path) -> list:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Image quality gate (pre-inference filter).")
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--config", type=Path, default=None, help="Override config YAML.")
    parser.add_argument("--no-strip-exif", action="store_false", dest="strip_exif",
                        help="Do not strip EXIF (inspection only).")
    parser.add_argument("--no-vehicle-check", action="store_false", dest="check_vehicle",
                        help="Skip the YOLO vehicle-presence check.")
    args = parser.parse_args()

    config = load_config(args.config)
    images = _find_images(args.source)
    if not images:
        log.error("No images found in: %s", args.source)
        return

    results = []
    for img_path in images:
        try:
            verdict = assess_quality(
                img_path, config=config,
                strip_exif=args.strip_exif, check_vehicle=args.check_vehicle,
            )
        except (FileNotFoundError, ValueError) as exc:
            verdict = {"valid": False, "problems": [f"unreadable: {exc}"], "scores": {}, "exif_removed": False}
        verdict["image"] = img_path.name
        results.append(verdict)
        status = "VALID" if verdict["valid"] else "INVALID"
        log.info("[%s] %s %s", status, img_path.name, verdict["problems"] or "")

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
