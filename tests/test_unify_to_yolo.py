"""
Tests for scripts/unify_to_yolo.py — the leakage-free grouped split + pHash dedup
(the core of the VehiDE-4 data fix) and the COCO→YOLO normalization guard.

These tests need only numpy + Pillow (for the dedup image path) — no torch/cv2 — so
they run in a light environment. Run:
  uv run --with pyyaml --with rich --with numpy --with pillow --with pytest \
      python -m pytest tests/test_unify_to_yolo.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import unify_to_yolo as u  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────

def _coco(images, anns):
    """Build a minimal COCO dict. images: [(id, file_name)], anns: [(image_id, cat)]."""
    return {
        "images": [{"id": i, "file_name": fn, "width": 640, "height": 480} for i, fn in images],
        "annotations": [
            {"id": k, "image_id": img_id, "category_id": cat, "segmentation": [], "area": 500}
            for k, (img_id, cat) in enumerate(anns)
        ],
    }


def _split_of(splits):
    return {img_id: s for s, ids in splits.items() for img_id in ids}


# ── Grouped split: same-vehicle photos never cross splits ─────────────

def test_filename_groups_stay_together_and_no_leakage():
    # 12 vehicles, 3 photos each (car{n}_1.._3) → grouping must keep each trio together.
    images, anns = [], []
    iid = 0
    for v in range(12):
        for shot in range(1, 4):
            images.append((iid, f"car{v:03d}_{shot}.jpg"))
            anns.append((iid, v % 4))  # spread across the 4 classes
            iid += 1

    coco = _coco(images, anns)
    splits, audit = u.create_grouped_splits(
        coco, images_dir=None, seed=42,
        group_from_filename=r"^(.*?)(?:[_-]?\d+)?$", dedup_phash=False,
    )

    assert audit["cross_split_groups"] == 0           # the headline guarantee
    # every car's 3 photos land in exactly one split
    where = _split_of(splits)
    for v in range(12):
        trio = [where[i] for i, fn in images if fn.startswith(f"car{v:03d}_")]
        assert len(set(trio)) == 1, f"car{v:03d} leaked across {set(trio)}"
    # all images placed exactly once
    total = sum(len(ids) for ids in splits.values())
    assert total == len(images)
    assert len({i for ids in splits.values() for i in ids}) == len(images)


def test_ratios_roughly_respected_with_independent_groups():
    # 100 independent images, no filename grouping (group_from_filename=None) →
    # each image is its own group → ~70/20/10.
    images = [(i, f"veh{i:04d}.jpg") for i in range(100)]
    anns = [(i, 0) for i in range(100)]
    coco = _coco(images, anns)
    splits, audit = u.create_grouped_splits(
        coco, images_dir=None, seed=0,
        group_from_filename=None, dedup_phash=False,
    )
    assert audit["cross_split_groups"] == 0
    assert audit["n_groups"] == 100
    assert 60 <= len(splits["train"]) <= 80
    assert 12 <= len(splits["val"]) <= 28
    assert len(splits["test"]) >= 1


def test_shared_prefix_regex_collapses_groups_documented_footgun():
    # Documents WHY the default is null: a digit-stripping regex on files that
    # share an alphabetic prefix collapses everything into ONE group.
    images = [(i, f"veh{i:04d}.jpg") for i in range(20)]
    anns = [(i, 0) for i in range(20)]
    coco = _coco(images, anns)
    _, audit = u.create_grouped_splits(
        coco, images_dir=None, seed=0,
        group_from_filename=r"^(.*?)(?:[_-]?\d+)?$", dedup_phash=False,
    )
    assert audit["n_groups"] == 1  # all collapse to key "veh" → would starve val/test


# ── Perceptual-hash dedup: near-duplicates merge even with distinct names ──

def _save(path: Path, arr: np.ndarray):
    from PIL import Image
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_phash_merges_near_duplicates_across_distinct_filenames(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # A horizontal gradient and a near-identical copy (mild noise) → same dHash.
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    grad = np.stack([grad] * 3, axis=-1)
    near = np.clip(grad.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    # A clearly different image: vertical gradient.
    vert = np.tile(np.linspace(0, 255, 64, dtype=np.uint8).reshape(64, 1), (1, 64))
    vert = np.stack([vert] * 3, axis=-1)

    _save(images_dir / "alpha.jpg", grad)
    _save(images_dir / "beta.jpg", near)   # near-dup of alpha, unrelated name
    _save(images_dir / "gamma.jpg", vert)  # distinct

    coco = _coco([(0, "alpha.jpg"), (1, "beta.jpg"), (2, "gamma.jpg")],
                 [(0, 0), (1, 0), (2, 1)])
    splits, audit = u.create_grouped_splits(
        coco, images_dir=images_dir, seed=1,
        group_from_filename=None, dedup_phash=True, phash_threshold=6,
    )

    assert audit["phash"]["computed"] == 3
    assert audit["phash"]["dup_pairs"] >= 1
    assert audit["cross_split_groups"] == 0
    where = _split_of(splits)
    assert where[0] == where[1]           # alpha & beta (near-dups) co-located
    assert audit["n_groups"] == 2         # {alpha,beta} and {gamma}


def test_missing_images_degrade_gracefully_to_filename_grouping(tmp_path):
    # dedup on but no image files present → counted as missing, no crash.
    coco = _coco([(0, "a_1.jpg"), (1, "a_2.jpg"), (2, "b_1.jpg")], [(0, 0), (1, 0), (2, 1)])
    splits, audit = u.create_grouped_splits(
        coco, images_dir=tmp_path, seed=2,
        group_from_filename=r"^(.*?)(?:[_-]?\d+)?$", dedup_phash=True,
    )
    assert audit["phash"]["missing"] == 3
    assert audit["cross_split_groups"] == 0
    where = _split_of(splits)
    assert where[0] == where[1]  # a_1 & a_2 grouped by filename


# ── dHash / Hamming primitives ────────────────────────────────────────

def test_hamming_distance():
    assert u._hamming(0b1010, 0b1010) == 0
    assert u._hamming(0b1010, 0b1000) == 1
    assert u._hamming(0b1111, 0b0000) == 4


# ── COCO→YOLO normalization guard ─────────────────────────────────────

def test_segmentation_guard_rejects_zero_dimensions():
    poly = [[10, 10, 100, 10, 100, 100, 10, 100]]
    assert u.coco_segmentation_to_yolo(poly, 0, 480) == []
    assert u.coco_segmentation_to_yolo(poly, 640, 0) == []


def test_segmentation_normalizes_in_unit_range():
    poly = [[0, 0, 640, 0, 640, 480, 0, 480]]
    out = u.coco_segmentation_to_yolo(poly, 640, 480)
    assert len(out) == 1
    assert all(0.0 <= v <= 1.0 for v in out[0])
