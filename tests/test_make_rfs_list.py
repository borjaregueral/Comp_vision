"""
Tests for scripts/make_rfs_list.py — the Repeat-Factor Sampling math (Tier 1.4).

Pure functions over {image: set(class_ids)}, no files/torch/cv2. Run:
  ./venv/bin/python -m pytest tests/test_make_rfs_list.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_rfs_list as r  # noqa: E402


def _dataset(common=90, rare=10):
    """`common` images with class 0 (frequent), `rare` with class 3 (scarce)."""
    d = {f"common_{i}.jpg": {0} for i in range(common)}
    d.update({f"rare_{i}.jpg": {3} for i in range(rare)})
    return d


# ── frequencies & per-class repeat factor ─────────────────────────────

def test_class_image_frequencies():
    f = r.class_image_frequencies(_dataset(90, 10))
    assert abs(f[0] - 0.9) < 1e-9
    assert abs(f[3] - 0.1) < 1e-9


def test_repeat_factor_oversamples_only_below_threshold():
    f = r.class_image_frequencies(_dataset(90, 10))
    rc = r.repeat_factor_per_class(f, t=0.4)
    assert abs(rc[3] - 2.0) < 1e-9     # sqrt(0.4/0.1) = 2 → rare class oversampled
    assert rc[0] == 1.0                # f(0)=0.9 > t → clamped to 1 (no oversample)


def test_image_factor_takes_the_rarest_class():
    rc = {0: 1.0, 3: 2.0}
    ri = r.repeat_factor_per_image({"a.jpg": {0, 3}, "b.jpg": {0}, "c.jpg": set()}, rc)
    assert ri["a.jpg"] == 2.0   # max(1, 2)
    assert ri["b.jpg"] == 1.0
    assert ri["c.jpg"] == 1.0   # no labels → 1


# ── stochastic rounding ───────────────────────────────────────────────

def test_stochastic_count():
    assert r.stochastic_count(2.0, 0.99) == 2     # integer factor → deterministic
    assert r.stochastic_count(1.0, 0.0) == 1
    assert r.stochastic_count(2.3, 0.2) == 3       # u < frac(0.3) → round up
    assert r.stochastic_count(2.3, 0.5) == 2       # u >= frac → round down


# ── end-to-end list build ─────────────────────────────────────────────

def test_build_rfs_list_repeats_rare_images_and_is_deterministic():
    ds = _dataset(90, 10)
    lines, audit = r.build_rfs_list(ds, t=0.4, seed=0)
    # integer factors (1 and 2) → exactly 90*1 + 10*2 = 110 lines
    assert audit["n_images"] == 100
    assert audit["n_lines"] == 110
    assert audit["oversample_ratio"] == 1.1
    # every rare image appears twice, every common once
    assert sum(1 for ln in lines if ln.startswith("rare_")) == 20
    assert sum(1 for ln in lines if ln.startswith("common_")) == 90
    # deterministic given the seed
    lines2, _ = r.build_rfs_list(ds, t=0.4, seed=0)
    assert lines == lines2


def test_no_oversampling_when_t_below_all_frequencies():
    ds = _dataset(90, 10)
    lines, audit = r.build_rfs_list(ds, t=0.001, seed=0)  # t < f(3)=0.1 → all r=1
    assert audit["n_lines"] == audit["n_images"] == 100
    assert audit["oversample_ratio"] == 1.0
