"""
Tests for scripts/business_metrics.py (T3.3).

Per-metric checks on hand-built records plus the plan requirement: on a synthetic
golden set of 20 cases every metric computes without error and returns values in
sensible ranges. Also checks the record adapter and the self-contained HTML report.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import business_metrics as bm  # noqa: E402


def _rec(lane="verde", est=300.0, real=300.0, sys_struct=False, gt_struct=False,
         sev_pred="leve", sev_gt="leve", ms=100, claim="C"):
    return {
        "claim_id": claim, "lane": lane, "estimated_eur": est,
        "structural_suspected": sys_struct, "severity_pred": sev_pred,
        "processing_time_ms": ms,
        "ground_truth": {"importe_real": real, "es_estructural": gt_struct,
                         "severidad_oficial": sev_gt},
    }


def _synthetic(n=20):
    rng = random.Random(123)
    sevs = ["leve", "moderado", "severo"]
    recs = []
    for i in range(n):
        real = rng.choice([200.0, 350.0, 600.0, 900.0, 1400.0, 1800.0])
        est = round(real * (1 + rng.uniform(-0.12, 0.12)), 2)
        lane = rng.choices(["verde", "ambar", "rojo"], weights=[5, 3, 2])[0]
        gt_struct = (i % 7 == 0)
        sys_struct = gt_struct if rng.random() < 0.9 else (not gt_struct)
        sev_gt = sevs[i % 3]
        sev_pred = sev_gt if rng.random() < 0.8 else rng.choice(sevs)
        recs.append(_rec(lane, est, real, sys_struct, gt_struct, sev_pred, sev_gt,
                         rng.randint(80, 400), f"C{i}"))
    return recs


CONFIG = bm.load_config()


# ── MAE ───────────────────────────────────────────────────────────────

def test_mae_euros_simple():
    recs = [_rec(lane="verde", est=100, real=120), _rec(lane="verde", est=200, real=180)]
    assert bm.mae_euros(recs, ["verde", "ambar"]) == 20.0


def test_mae_excludes_rojo():
    recs = [_rec(lane="verde", est=100, real=110), _rec(lane="rojo", est=100, real=2000)]
    assert bm.mae_euros(recs, ["verde", "ambar"]) == 10.0  # rojo not counted


# ── % green ───────────────────────────────────────────────────────────

def test_pct_green():
    recs = [_rec(lane="verde"), _rec(lane="verde"), _rec(lane="ambar"), _rec(lane="rojo")]
    g = bm.pct_green(recs)
    assert g["over_total"] == 50.0
    assert g["over_liquidables"] == round(100 * 2 / 3, 2)


# ── FN structural ─────────────────────────────────────────────────────

def test_fn_structural_counts_missed():
    recs = [
        _rec(gt_struct=True, sys_struct=True, lane="rojo"),     # caught
        _rec(gt_struct=True, sys_struct=False, lane="ambar"),   # MISSED
        _rec(gt_struct=False, sys_struct=False, lane="verde"),  # not positive
    ]
    fn = bm.fn_rate_structural(recs)
    assert fn["n_positives"] == 2 and fn["n_missed"] == 1
    assert fn["pct"] == 50.0


def test_fn_not_counted_when_sent_to_red():
    recs = [_rec(gt_struct=True, sys_struct=False, lane="rojo")]  # missed flag but red anyway
    assert bm.fn_rate_structural(recs)["n_missed"] == 0


def test_fn_none_without_positives():
    assert bm.fn_rate_structural([_rec(gt_struct=False)])["pct"] is None


# ── kappa & within ────────────────────────────────────────────────────

def test_weighted_kappa_perfect_agreement():
    recs = [_rec(sev_pred=s, sev_gt=s) for s in ["leve", "moderado", "severo", "leve"]]
    assert bm.weighted_kappa(recs, ["leve", "moderado", "severo"]) == 1.0


def test_within_15pct():
    recs = [_rec(lane="verde", est=100, real=100),   # 0% error → within
            _rec(lane="verde", est=100, real=120)]   # 16.7% error → out
    assert bm.pct_within(recs, ["verde", "ambar"], 0.15) == 50.0


# ── Adapter ───────────────────────────────────────────────────────────

def test_record_from_output():
    output = {
        "claim_id": "SIN-1", "lane": "rojo",
        "estimacion": {"total_eur": 540.0},
        "damages": [
            {"severity": "leve", "structural_suspicion": False},
            {"severity": "severo", "structural_suspicion": True},
        ],
        "audit": {"processing_time_ms": 321},
    }
    gt = {"importe_real": 500.0, "es_estructural": True, "severidad_oficial": "severo"}
    rec = bm.record_from_output(output, gt)
    assert rec["severity_pred"] == "severo"        # max of damages
    assert rec["structural_suspected"] is True      # any structural
    assert rec["estimated_eur"] == 540.0
    assert rec["processing_time_ms"] == 321


# ── Aggregate on 20 synthetic cases (plan requirement) ────────────────

def test_compute_all_metrics_synthetic_20():
    recs = _synthetic(20)
    m = bm.compute_all_metrics(recs, CONFIG)
    assert m["n_cases"] == 20
    assert m["mae_eur"]["point"] >= 0
    assert 0 <= m["pct_green"]["over_total"] <= 100
    assert m["fn_structural_pct"]["pct"] is not None  # we seeded structural positives
    assert -1.0 <= m["weighted_kappa"] <= 1.0
    assert 0 <= m["within_15pct"]["point"] <= 100
    assert m["mean_processing_ms"]["point"] > 0
    # bootstrap CI brackets the point estimate.
    lo, hi = m["mae_eur"]["ci"]
    assert lo <= m["mae_eur"]["point"] <= hi
    # all tramos present
    assert set(m["mae_by_tramo"]) == {"<500", "500-1500", ">1500"}


def test_targets_pass_evaluation_present():
    recs = [_rec(lane="verde", est=100, real=100, sev_pred="leve", sev_gt="leve")]
    recs.append(_rec(lane="verde", est=200, real=205, sev_pred="severo", sev_gt="severo"))
    m = bm.compute_all_metrics(recs, CONFIG)
    assert "mae_eur" in m["pass"]
    assert m["pass"]["mae_eur"] is True  # tiny errors → under the 120€ target


# ── HTML report ───────────────────────────────────────────────────────

def test_generate_html_report_self_contained(tmp_path):
    recs = _synthetic(20)
    m = bm.compute_all_metrics(recs, CONFIG)
    out = bm.generate_html_report(m, tmp_path / "report.html", fecha="2026-06-06", records=recs)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Métricas de negocio" in html
    assert "MAE" in html
    assert "data:image/png;base64," in html      # charts embedded
    assert 'src="http' not in html               # no external assets
