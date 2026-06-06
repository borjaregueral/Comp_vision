"""
Tests for scripts/calibrate_confidence.py (T3.4).

Core requirement: calibrated confidences have a Brier score <= the raw ones on a
held-out set. Plus the in-sample guarantee, save/load roundtrip, Platt path,
reliability curve / ECE, and error cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calibrate_confidence as cc  # noqa: E402


def _miscalibrated(n, seed):
    """Overconfident detector: high raw confidence but lower true accuracy."""
    rng = np.random.default_rng(seed)
    levels = np.array([0.5, 0.7, 0.9, 0.95])
    true_acc = {0.5: 0.45, 0.7: 0.5, 0.9: 0.55, 0.95: 0.6}
    conf = rng.choice(levels, size=n)
    labels = np.array([1.0 if rng.random() < true_acc[c] else 0.0 for c in conf])
    return conf, labels


# ── Brier improvement (the plan requirement) ─────────────────────────

def test_isotonic_improves_brier_on_holdout():
    conf, labels = _miscalibrated(1000, seed=1)
    tr, va = slice(0, 600), slice(600, None)
    cal = cc.ConfidenceCalibrator("isotonic").fit(conf[tr], labels[tr])
    brier_raw = cc.brier_score(conf[va], labels[va])
    brier_cal = cc.brier_score(cal.transform(conf[va]), labels[va])
    assert brier_cal <= brier_raw


def test_calibration_not_worse_in_sample():
    conf, labels = _miscalibrated(800, seed=2)
    cal = cc.ConfidenceCalibrator("isotonic").fit(conf, labels)
    assert cc.brier_score(cal.transform(conf), labels) <= cc.brier_score(conf, labels) + 1e-9


# ── Persistence ───────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    conf, labels = _miscalibrated(300, seed=3)
    cal = cc.ConfidenceCalibrator("isotonic").fit(conf, labels)
    path = cc.save_calibrator(cal, tmp_path / "models" / "baseline_v1.0" / "confidence_calibrator.pkl")
    assert path.exists()
    loaded = cc.load_calibrator(path)
    sample = np.array([0.3, 0.6, 0.9, 0.95])
    assert np.allclose(loaded.transform(sample), cal.transform(sample))


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cc.load_calibrator(tmp_path / "nope.pkl")


# ── Platt & metrics ───────────────────────────────────────────────────

def test_platt_outputs_probabilities():
    conf, labels = _miscalibrated(400, seed=4)
    cal = cc.ConfidenceCalibrator("platt").fit(conf, labels)
    out = cal.transform(np.array([0.1, 0.5, 0.99]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_brier_score_value():
    assert cc.brier_score([0.9, 0.1], [1, 0]) == pytest.approx(0.01, abs=1e-9)


def test_reliability_curve_counts_sum_to_n():
    conf, labels = _miscalibrated(200, seed=5)
    curve = cc.reliability_curve(conf, labels, n_bins=10)
    assert sum(curve["counts"]) == 200
    assert len(curve["bin_centers"]) == 10


def test_ece_zero_for_well_calibrated():
    conf = [0.5] * 100
    labels = [1] * 50 + [0] * 50  # bin 0.5 has accuracy 0.5 == confidence
    assert cc.expected_calibration_error(conf, labels, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_plot_reliability_writes_png(tmp_path):
    conf, labels = _miscalibrated(200, seed=6)
    out = cc.plot_reliability(conf, labels, tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0


# ── Errors ────────────────────────────────────────────────────────────

def test_unknown_method_raises():
    with pytest.raises(ValueError):
        cc.ConfidenceCalibrator("magic")


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        cc.ConfidenceCalibrator("isotonic").transform([0.5])
