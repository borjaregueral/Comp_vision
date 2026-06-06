#!/usr/bin/env python3
"""
business_metrics.py — Business metrics against the golden set (T3.3).

The PRIMARY metric is MAE in euros, not mAP. This module compares the system's
output to the human ground truth and reports, with bootstrap confidence
intervals (we report ranges, not just point estimates):

- MAE € (on liquidable lanes verde+ámbar) and MAE by importe tramo
- % of cases in the green lane (over total and over liquidables)
- false-negative rate on suspected structural damage (the non-negotiable ≤2%)
- Cohen's weighted kappa of severity (system vs surveyor)
- % of estimates within ±15% of the real amount
- mean processing time per claim

Targets are shown as pass/fail only; they never change the computed values.

Records contract
----------------
Each record (one per claim):
    {
      "claim_id": str,
      "lane": "verde"|"ambar"|"rojo",
      "estimated_eur": float,
      "structural_suspected": bool,        # system flag
      "severity_pred": "leve"|"moderado"|"severo"|None,
      "processing_time_ms": int,
      "ground_truth": {
          "importe_real": float,
          "es_estructural": bool,
          "severidad_oficial": "leve"|"moderado"|"severo",
      },
    }

Public API
----------
    load_config(path=None) -> dict
    record_from_output(output, ground_truth, processing_time_ms=None) -> dict
    compute_all_metrics(records, config=None) -> dict
    generate_html_report(metrics, output_path, fecha, records=None) -> Path
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("business_metrics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "business_metrics.yaml"

_SEV_RANK = {"leve": 0, "moderado": 1, "severo": 2}


def load_config(path: Optional[Path] = None) -> dict:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Business-metrics config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Adapter from a full inference output ─────────────────────────────

def record_from_output(output: dict, ground_truth: dict,
                       processing_time_ms: Optional[int] = None) -> dict:
    """Build a metrics record from a validated inference output + its ground truth."""
    damages = output.get("damages", []) or []
    severities = [d.get("severity") for d in damages if d.get("severity") in _SEV_RANK]
    severity_pred = max(severities, key=lambda s: _SEV_RANK[s]) if severities else None
    structural = any(bool(d.get("structural_suspicion")) for d in damages)
    audit = output.get("audit", {}) or {}
    return {
        "claim_id": output.get("claim_id"),
        "lane": output.get("lane"),
        "estimated_eur": (output.get("estimacion") or {}).get("total_eur"),
        "structural_suspected": structural,
        "severity_pred": severity_pred,
        "processing_time_ms": processing_time_ms if processing_time_ms is not None
        else audit.get("processing_time_ms"),
        "ground_truth": ground_truth,
    }


# ── Bootstrap ────────────────────────────────────────────────────────

def _bootstrap_ci(items, stat_fn, n_resamples, seed, alpha):
    """Percentile bootstrap CI for stat_fn over a list of items. Deterministic (seed)."""
    items = list(items)
    if len(items) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    size = len(items)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, size, size)
        sample = [items[i] for i in idx]
        try:
            stats.append(stat_fn(sample))
        except Exception:  # pragma: no cover - a degenerate resample is just skipped
            continue
    if not stats:
        return [None, None]
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return [round(lo, 3), round(hi, 3)]


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ── Individual metrics ───────────────────────────────────────────────

def _liquidable(records, lanes):
    return [r for r in records if r.get("lane") in lanes
            and (r.get("ground_truth") or {}).get("importe_real") is not None
            and r.get("estimated_eur") is not None]

def _abs_errors(records):
    return [abs(r["estimated_eur"] - r["ground_truth"]["importe_real"]) for r in records]


def mae_euros(records, lanes):
    errs = _abs_errors(_liquidable(records, lanes))
    return round(_mean(errs), 2) if errs else None


def mae_by_tramo(records, lanes, tramos):
    out = {}
    liq = _liquidable(records, lanes)
    for tramo in tramos:
        name, cap = tramo["name"], tramo["max"]
        prev = 0
        # determine lower bound from order of tramos
        members = []
        for r in liq:
            real = r["ground_truth"]["importe_real"]
            if cap is None:
                if real >= _tramo_lower(tramos, name):
                    members.append(r)
            elif _tramo_lower(tramos, name) <= real < cap:
                members.append(r)
        errs = _abs_errors(members)
        out[name] = {"mae_eur": round(_mean(errs), 2) if errs else None, "n": len(members)}
    return out


def _tramo_lower(tramos, name):
    lower = 0
    for tramo in tramos:
        if tramo["name"] == name:
            return lower
        lower = tramo["max"] if tramo["max"] is not None else lower
    return 0


def pct_green(records):
    total = len(records)
    liquidables = [r for r in records if r.get("lane") in ("verde", "ambar")]
    n_green = sum(1 for r in records if r.get("lane") == "verde")
    return {
        "over_total": round(100 * n_green / total, 2) if total else None,
        "over_liquidables": round(100 * n_green / len(liquidables), 2) if liquidables else None,
        "n_green": n_green,
        "n_total": total,
    }


def fn_rate_structural(records):
    """FN = ground-truth structural that the system did NOT flag (and not sent to red)."""
    positives = [r for r in records if (r.get("ground_truth") or {}).get("es_estructural")]
    if not positives:
        return {"pct": None, "n_positives": 0, "n_missed": 0}
    missed = [r for r in positives
              if not r.get("structural_suspected") and r.get("lane") != "rojo"]
    return {"pct": round(100 * len(missed) / len(positives), 2),
            "n_positives": len(positives), "n_missed": len(missed)}


def weighted_kappa(records, labels):
    from sklearn.metrics import cohen_kappa_score

    pairs = [(r.get("severity_pred"), (r.get("ground_truth") or {}).get("severidad_oficial"))
             for r in records]
    pairs = [(p, t) for p, t in pairs if p in _SEV_RANK and t in _SEV_RANK]
    if len(pairs) < 2 or len({t for _, t in pairs}) < 2:
        return None  # kappa undefined with <2 classes
    pred = [p for p, _ in pairs]
    true = [t for _, t in pairs]
    return round(float(cohen_kappa_score(pred, true, labels=labels, weights="quadratic")), 4)


def pct_within(records, lanes, within):
    liq = _liquidable(records, lanes)
    if not liq:
        return None
    hits = sum(1 for r in liq
               if r["ground_truth"]["importe_real"] > 0
               and abs(r["estimated_eur"] - r["ground_truth"]["importe_real"])
               / r["ground_truth"]["importe_real"] <= within)
    return round(100 * hits / len(liq), 2)


def mean_processing_ms(records):
    times = [r["processing_time_ms"] for r in records if r.get("processing_time_ms") is not None]
    return round(_mean(times), 1) if times else None


# ── Aggregate ────────────────────────────────────────────────────────

def compute_all_metrics(records: list, config: Optional[dict] = None) -> dict:
    """Compute every business metric with bootstrap CIs. Pure (no I/O besides config)."""
    config = config if config is not None else load_config()
    lanes = config.get("mae_lanes", ["verde", "ambar"])
    bs = config.get("bootstrap", {})
    n_bs, seed, alpha = bs.get("n_resamples", 2000), bs.get("seed", 42), bs.get("alpha", 0.05)
    within = config.get("within_pct", 0.15)
    labels = config.get("severity_labels", ["leve", "moderado", "severo"])
    tramos = config.get("importe_tramos", [])

    liq = _liquidable(records, lanes)
    abs_errs = _abs_errors(liq)

    mae = {"point": round(_mean(abs_errs), 2) if abs_errs else None, "n": len(liq),
           "ci": _bootstrap_ci(abs_errs, _mean, n_bs, seed, alpha) if abs_errs else [None, None]}

    green = pct_green(records)
    green_flags = [1 if r.get("lane") == "verde" else 0 for r in records]
    green["ci_over_total"] = _bootstrap_ci(green_flags, lambda xs: 100 * _mean(xs), n_bs, seed, alpha)

    fn = fn_rate_structural(records)

    within_flags = [
        1 if (r["ground_truth"]["importe_real"] > 0 and
              abs(r["estimated_eur"] - r["ground_truth"]["importe_real"])
              / r["ground_truth"]["importe_real"] <= within) else 0
        for r in liq
    ]
    within_m = {"point": pct_within(records, lanes, within),
                "ci": _bootstrap_ci(within_flags, lambda xs: 100 * _mean(xs), n_bs, seed, alpha)
                if within_flags else [None, None]}

    times = [r["processing_time_ms"] for r in records if r.get("processing_time_ms") is not None]
    proc = {"point": mean_processing_ms(records),
            "ci": _bootstrap_ci(times, _mean, n_bs, seed, alpha) if times else [None, None]}

    targets = config.get("targets", {})
    kappa = weighted_kappa(records, labels)
    metrics = {
        "n_cases": len(records),
        "mae_eur": mae,
        "mae_by_tramo": mae_by_tramo(records, lanes, tramos),
        "pct_green": green,
        "fn_structural_pct": fn,
        "weighted_kappa": kappa,
        "within_15pct": within_m,
        "mean_processing_ms": proc,
        "targets": targets,
    }
    metrics["pass"] = _evaluate_targets(metrics, targets)
    return metrics


def _evaluate_targets(metrics, targets) -> dict:
    result = {}
    if metrics["mae_eur"]["point"] is not None and "mae_eur_max" in targets:
        result["mae_eur"] = metrics["mae_eur"]["point"] <= targets["mae_eur_max"]
    if metrics["pct_green"]["over_total"] is not None and "pct_green_min" in targets:
        result["pct_green"] = metrics["pct_green"]["over_total"] >= targets["pct_green_min"]
    if metrics["fn_structural_pct"]["pct"] is not None and "fn_structural_max_pct" in targets:
        result["fn_structural"] = metrics["fn_structural_pct"]["pct"] <= targets["fn_structural_max_pct"]
    if metrics["weighted_kappa"] is not None and "weighted_kappa_min" in targets:
        result["weighted_kappa"] = metrics["weighted_kappa"] >= targets["weighted_kappa_min"]
    if metrics["within_15pct"]["point"] is not None and "within_15pct_min" in targets:
        result["within_15pct"] = metrics["within_15pct"]["point"] >= targets["within_15pct_min"]
    return result


# ── HTML report ──────────────────────────────────────────────────────

def _fig_to_base64(fig):
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_html_report(metrics: dict, output_path, fecha: str, records: Optional[list] = None) -> Path:
    """Write a self-contained HTML business report (tables + CIs + embedded charts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    charts_html = ""
    if records:
        liq = _liquidable(records, metrics.get("targets", {}) and ["verde", "ambar"])
        if liq:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(_abs_errors(liq), bins=10, color="#4488FF", edgecolor="white")
            ax.set_title("Distribución del error absoluto (€)")
            ax.set_xlabel("|estimado - real| (€)")
            ax.set_ylabel("nº casos")
            charts_html += f'<img alt="error histogram" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'
            plt.close(fig)

        lanes = {}
        for r in records:
            lanes[r.get("lane")] = lanes.get(r.get("lane"), 0) + 1
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = {"verde": "#2ecc71", "ambar": "#f1c40f", "rojo": "#e74c3c"}
        ax.bar(list(lanes), list(lanes.values()),
               color=[colors.get(k, "#888") for k in lanes])
        ax.set_title("Distribución por carril")
        ax.set_ylabel("nº casos")
        charts_html += f'<img alt="lane distribution" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'
        plt.close(fig)

    def _pf(key):
        p = metrics.get("pass", {}).get(key)
        return "" if p is None else (" ✅" if p else " ❌")

    def _ci(ci):
        if not ci or ci[0] is None:
            return "—"
        return f"[{ci[0]}, {ci[1]}]"

    mae = metrics["mae_eur"]
    green = metrics["pct_green"]
    fn = metrics["fn_structural_pct"]
    within = metrics["within_15pct"]
    proc = metrics["mean_processing_ms"]

    rows = [
        ("MAE € (verde+ámbar)", mae["point"], _ci(mae["ci"]), f"≤ {metrics['targets'].get('mae_eur_max','-')}", _pf("mae_eur")),
        ("% carril verde (total)", green["over_total"], _ci(green.get("ci_over_total")), f"≥ {metrics['targets'].get('pct_green_min','-')}", _pf("pct_green")),
        ("FN estructural (%)", fn["pct"], "—", f"≤ {metrics['targets'].get('fn_structural_max_pct','-')}", _pf("fn_structural")),
        ("Cohen weighted kappa", metrics["weighted_kappa"], "—", f"≥ {metrics['targets'].get('weighted_kappa_min','-')}", _pf("weighted_kappa")),
        ("% dentro de ±15%", within["point"], _ci(within["ci"]), f"≥ {metrics['targets'].get('within_15pct_min','-')}", _pf("within_15pct")),
        ("Tiempo medio (ms)", proc["point"], _ci(proc["ci"]), "—", ""),
    ]
    rows_html = "\n".join(
        f"<tr><td>{n}</td><td>{v}</td><td>{ci}</td><td>{t}</td><td>{pf}</td></tr>"
        for n, v, ci, t, pf in rows
    )
    tramo_rows = "\n".join(
        f"<tr><td>{name}</td><td>{d['mae_eur']}</td><td>{d['n']}</td></tr>"
        for name, d in metrics["mae_by_tramo"].items()
    )

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Métricas de negocio — {fecha}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 1.4rem; }} table {{ border-collapse: collapse; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
 th {{ background: #f4f4f4; }} img {{ margin: 8px; vertical-align: top; }}
 .note {{ color: #666; font-size: 0.85rem; }}
</style></head><body>
<h1>Métricas de negocio — {fecha}</h1>
<p class="note">N = {metrics['n_cases']} casos. Métrica primaria: MAE en €. IC del 95% por bootstrap.
Los objetivos son orientativos (pass/fail), no alteran el cálculo.</p>
<table>
<tr><th>Métrica</th><th>Valor</th><th>IC 95%</th><th>Objetivo</th><th></th></tr>
{rows_html}
</table>
<h2>MAE por tramo de importe</h2>
<table><tr><th>Tramo (€ real)</th><th>MAE €</th><th>n</th></tr>
{tramo_rows}
</table>
<h2>Gráficos</h2>
{charts_html or '<p class="note">Sin registros para graficar.</p>'}
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    log.info("Business report written to %s", output_path)
    return output_path
