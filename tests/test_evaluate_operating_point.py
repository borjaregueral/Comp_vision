"""
Tests for the Tier 1.6 operating-point analysis in scripts/evaluate.py.

analyze_operating_point() only does numpy math over the val P/R/F1 curves, so we
feed it a synthetic metrics stub (no model, no GPU, no real val run) and check that
the recall-biased point (F-beta, beta=2) sits at a LOWER confidence — i.e. higher
recall — than the plain F1 peak. Run:
  ./venv/bin/python -m pytest tests/test_evaluate_operating_point.py -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate as e  # noqa: E402  (imports cv2/matplotlib; present in the train venv)


class _Stub:
    """Bare attribute holder to mimic Ultralytics' seg Metric / SegmentMetrics."""


def _metrics_with_classic_curves(n: int = 200, nc: int = 1):
    """Classic shapes: recall falls and precision rises with confidence.

    r(conf) = 1 - conf, p(conf) = conf → F1 peaks at conf 0.5; F-beta(beta=2),
    which weights recall double, peaks at a LOWER conf.
    """
    px = np.linspace(0.0, 1.0, n)
    r = np.tile(1.0 - px, (nc, 1))
    p = np.tile(px, (nc, 1))
    f1 = 2 * p * r / (p + r + 1e-9)
    seg = _Stub()
    seg.px, seg.p_curve, seg.r_curve, seg.f1_curve = px, p, r, f1
    m = _Stub()
    m.seg = seg
    return m


def test_recall_biased_point_has_lower_conf_and_higher_recall_than_f1_peak():
    op = e.analyze_operating_point(_metrics_with_classic_curves(), beta=2.0)
    assert op["available"] is True
    # F1 peaks at conf ~0.5 for these symmetric curves
    assert abs(op["f1_peak"]["conf"] - 0.5) < 0.05
    # The whole point of Tier 1.6: recall-biased point trades precision for recall
    assert op["recall_biased"]["conf"] < op["f1_peak"]["conf"]
    assert op["recall_biased"]["recall"] > op["f1_peak"]["recall"]
    assert op["recall_biased"]["precision"] < op["f1_peak"]["precision"]


def test_higher_beta_pushes_conf_even_lower():
    # beta=4 weights recall more than beta=2 → operating conf should drop further.
    op2 = e.analyze_operating_point(_metrics_with_classic_curves(), beta=2.0)
    op4 = e.analyze_operating_point(_metrics_with_classic_curves(), beta=4.0)
    assert op4["recall_biased"]["conf"] <= op2["recall_biased"]["conf"]
    assert op4["recall_biased"]["recall"] >= op2["recall_biased"]["recall"]


def test_multiclass_curves_are_averaged():
    op = e.analyze_operating_point(_metrics_with_classic_curves(nc=4), beta=2.0)
    assert op["available"] is True
    assert 0.0 <= op["recall_biased"]["conf"] <= 1.0


def test_missing_or_empty_curves_degrade_gracefully():
    assert e.analyze_operating_point(_Stub(), beta=2.0)["available"] is False  # no .seg
    seg = _Stub()
    seg.px, seg.p_curve, seg.r_curve, seg.f1_curve = (
        np.array([]), np.array([]), np.array([]), np.array([])
    )
    m = _Stub()
    m.seg = seg
    assert e.analyze_operating_point(m, beta=2.0)["available"] is False
