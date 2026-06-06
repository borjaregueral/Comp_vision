#!/usr/bin/env python3
"""
calibrate_confidence.py — Confidence calibration without retraining (T3.4).

A detector's raw confidence is not a reliable probability. We fit a calibrator
(isotonic regression or Platt scaling) on (raw_confidence, was_correct) pairs so
that "confidence 0.85" really means ~85% accuracy — important because the green
lane threshold is 0.85.

The calibrator is fitted once (on golden/validation data, T3.5), pickled to
models/baseline_v1.0/confidence_calibrator.pkl, and applied at inference via
predict.py --calibrate. The model itself is never retrained.

Public API
----------
    load_config(path=None) -> dict
    ConfidenceCalibrator(method).fit(conf, labels).transform(conf)
    save_calibrator(cal, path) / load_calibrator(path)
    brier_score(probs, labels) -> float
    reliability_curve(conf, labels, n_bins) -> dict
    expected_calibration_error(conf, labels, n_bins) -> float
    plot_reliability(conf, labels, out_path, n_bins) -> Path
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("calibrate_confidence")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "calibration.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Calibration config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Calibrator ───────────────────────────────────────────────────────

class ConfidenceCalibrator:
    """Maps raw confidence → calibrated probability. method: 'isotonic' | 'platt'."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.model = None

    def fit(self, confidences, labels) -> "ConfidenceCalibrator":
        conf = np.asarray(confidences, dtype=float)
        lab = np.asarray(labels, dtype=float)
        if conf.size < 2:
            raise ValueError("Need at least 2 samples to fit a calibrator.")
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(conf, lab)
        else:  # platt
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            self.model.fit(conf.reshape(-1, 1), lab.astype(int))
        return self

    def transform(self, confidences):
        if self.model is None:
            raise RuntimeError("Calibrator is not fitted.")
        conf = np.asarray(confidences, dtype=float)
        if self.method == "isotonic":
            out = self.model.predict(conf)
        else:
            out = self.model.predict_proba(conf.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    __call__ = transform


def save_calibrator(calibrator: ConfidenceCalibrator, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(calibrator, fh)
    return path


def load_calibrator(path) -> ConfidenceCalibrator:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibrator not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── Metrics ──────────────────────────────────────────────────────────

def brier_score(probs, labels) -> float:
    """Mean squared error between probabilities and binary labels (lower = better)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def reliability_curve(confidences, labels, n_bins: int = 10) -> dict:
    """Per-bin mean confidence vs empirical accuracy (the reliability diagram)."""
    conf = np.asarray(confidences, dtype=float)
    lab = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    centers, accs, confs, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        centers.append(round(float((edges[b] + edges[b + 1]) / 2), 3))
        counts.append(c)
        accs.append(round(float(lab[mask].mean()), 4) if c else None)
        confs.append(round(float(conf[mask].mean()), 4) if c else None)
    return {"bin_centers": centers, "bin_accuracy": accs, "bin_confidence": confs, "counts": counts}


def expected_calibration_error(confidences, labels, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over bins."""
    curve = reliability_curve(confidences, labels, n_bins)
    total = sum(curve["counts"]) or 1
    ece = 0.0
    for acc, cf, n in zip(curve["bin_accuracy"], curve["bin_confidence"], curve["counts"]):
        if n and acc is not None and cf is not None:
            ece += (n / total) * abs(acc - cf)
    return round(ece, 4)


def plot_reliability(confidences, labels, out_path, n_bins: int = 10) -> Path:
    """Render the reliability diagram (raw) to a PNG file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = reliability_curve(confidences, labels, n_bins)
    xs = [c for c, a in zip(curve["bin_confidence"], curve["bin_accuracy"]) if a is not None]
    ys = [a for a in curve["bin_accuracy"] if a is not None]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfecta")
    ax.plot(xs, ys, "o-", color="#4488FF", label="modelo")
    ax.set_xlabel("Confianza media"); ax.set_ylabel("Acierto empírico")
    ax.set_title("Reliability diagram"); ax.legend()
    fig.savefig(out_path, bbox_inches="tight", dpi=90)
    plt.close(fig)
    return out_path
