#!/usr/bin/env python3
"""
triage.py — Deterministic green/amber/red lane assignment.

The model NEVER decides the lane. The model (and the rest of the pipeline)
produce inputs — quality, consolidated damages, cost estimate, alerts — and
THESE deterministic rules decide the lane. No LLM, no randomness (rule 10).

Design
------
The tunable / audit-relevant parts live in business_rules/lane_rules.yaml:
thresholds, stable rule IDs, evaluation order and human-readable reason
templates. The *predicates* (the actual condition logic) are vetted Python here,
bound to each rule by its stable ID. We deliberately do NOT ``eval`` the YAML
``condition`` strings (that would run arbitrary code from config); those strings
serve as readable documentation of what each coded predicate checks.

Evaluation order: RED first (any matching red rule wins), then GREEN (every
green condition must hold), otherwise AMBER by default.

Public API
----------
    load_rules(path=None) -> dict
    assign_lane(report, metadata, rules=None) -> (lane, rule_id, reason)

``lane`` is one of "verde" | "ambar" | "rojo" (matches the output schema enum).
``rule_id`` matches ^(VERDE|AMBAR|ROJO)-[0-9]+$.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("triage")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES = PROJECT_ROOT / "business_rules" / "lane_rules.yaml"


# ── Rules loading ────────────────────────────────────────────────────

def load_rules(path: Optional[Path] = None) -> dict:
    """Load the lane rules (thresholds, rule metadata, reason templates)."""
    import yaml

    rules_path = Path(path) if path else DEFAULT_RULES
    if not rules_path.exists():
        raise FileNotFoundError(f"Lane rules not found: {rules_path}")
    with open(rules_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _thresholds(rules: dict) -> dict:
    return rules.get("thresholds", {}) or {}


def _format(template: str, ctx: dict) -> str:
    """Fill a reason template, falling back to the raw template on bad keys."""
    if not template:
        return ""
    try:
        return template.format(**ctx)
    except (KeyError, IndexError, ValueError):
        return template


# ── RED predicates (bound by rule id) ────────────────────────────────

def _red_predicate(rule_id: str, report: dict, metadata: dict, th: dict) -> Tuple[bool, dict]:
    """Return (fired, context) for the red rule identified by ``rule_id``.

    Unknown ids return (False, {}) so adding a documented-but-unimplemented red
    rule to the YAML cannot silently send everything to red.
    """
    damages = report.get("damages") or []
    estim = report.get("estimacion") or {}
    alerts = report.get("alerts") or []
    quality = report.get("quality") or {}

    if rule_id == "ROJO-1":  # structural damage suspected
        hit = [d for d in damages if d.get("structural_suspicion")]
        parts = ", ".join(sorted({d.get("part", "zona no especificada") for d in hit}))
        return bool(hit), {"parts": parts or "zona no especificada"}

    if rule_id == "ROJO-2":  # estimate above auto-management ceiling
        total = estim.get("total_eur")
        limit = th.get("cost_red_min", 1500)
        return (total is not None and total > limit), {"total_eur": total, "limit": limit}

    if rule_id == "ROJO-3":  # critical fraud/inconsistency alert
        crit = [a for a in alerts if a.get("severity") == "critical"]
        descs = "; ".join(a.get("description", "alerta crítica") for a in crit)
        return bool(crit), {"alert_descriptions": descs}

    if rule_id == "ROJO-4":  # high-value vehicle
        valor = metadata.get("valor_vehiculo_estimado")
        limit = th.get("high_value_vehicle", 40000)
        return (valor is not None and valor > limit), {"valor": valor}

    if rule_id == "ROJO-5":  # suspicious claim history
        n = metadata.get("siniestros_12m", 0)
        return (n >= th.get("history_red_min", 4)), {"n": n}

    if rule_id == "ROJO-6":  # invalid quality AND image-manipulation alert
        not_valid = not quality.get("valid", True)
        manipulated = any(a.get("id") == "image_manipulation" for a in alerts)
        problems = sorted({
            p for img in quality.get("per_image", []) for p in img.get("problems", [])
        })
        return (not_valid and manipulated), {"problems": ", ".join(problems) or "calidad insuficiente"}

    return False, {}


# ── GREEN conditions ─────────────────────────────────────────────────

def _green_checks(report: dict, metadata: dict, th: dict) -> list:
    """Return [(criterion_name, passed)] for every VERDE-1 condition."""
    quality = report.get("quality") or {}
    estim = report.get("estimacion") or {}
    damages = report.get("damages") or []
    alerts = report.get("alerts") or []

    conf_min = th.get("confidence_green_min", 0.85)
    cost_max = th.get("cost_green_max", 800)
    hist_max = th.get("history_green_max", 2)

    total = estim.get("total_eur")
    conf_overall = estim.get("confidence_overall")
    siniestros = metadata.get("siniestros_12m", 0)

    return [
        ("calidad_valida", bool(quality.get("valid"))),
        (f"confianza_estimacion>={conf_min}", conf_overall is not None and conf_overall >= conf_min),
        (f"importe<{cost_max}€", total is not None and total < cost_max),
        ("sin_alertas_warning_o_critical",
         not any(a.get("severity") in ("warning", "critical") for a in alerts)),
        (f"siniestros_12m<={hist_max}", siniestros <= hist_max),
        (f"confianza_todos_los_daños>={conf_min}",
         all(d.get("confidence", 0.0) >= conf_min for d in damages)),
        ("sin_sospecha_estructural",
         not any(d.get("structural_suspicion") for d in damages)),
    ]


# ── Public entry point ───────────────────────────────────────────────

def assign_lane(report: dict, metadata: dict, rules: Optional[dict] = None) -> Tuple[str, str, str]:
    """Assign the operational lane for a consolidated claim report.

    Args:
        report: Consolidated report with keys quality, damages, estimacion,
            alerts (estimacion may be missing at this stage — handled safely:
            cost-based rules simply do not fire and the case cannot be green).
        metadata: Claim metadata (valor_vehiculo_estimado, siniestros_12m, ...).
        rules: Pre-loaded rules dict; loaded from YAML if None.

    Returns:
        (lane, rule_id, reason) where lane in {"verde", "ambar", "rojo"}.
    """
    rules = rules if rules is not None else load_rules()
    th = _thresholds(rules)

    # RED — first matching rule wins (evaluated in YAML order).
    for rule in rules.get("red", []) or []:
        rid = rule.get("id", "")
        fired, ctx = _red_predicate(rid, report, metadata, th)
        if fired:
            reason = _format(rule.get("reason_template", ""), ctx) or rule.get("description", rid)
            log.debug("triage decision: rojo (%s)", rid)
            return ("rojo", rid, reason)

    # AMBAR-2 — valid quality but NO damage detected (possible false negative).
    # Evaluated before green so a no-damage claim is never auto-resolved.
    quality = report.get("quality", {}) or {}
    if quality.get("valid") and len(report.get("damages", []) or []) == 0:
        nd = rules.get("no_damage_amber", {}) or {}
        reason = nd.get("reason_template", "") or nd.get("description", "")
        log.debug("triage decision: ambar (%s) — sin daño con calidad válida", nd.get("id", "AMBAR-2"))
        return ("ambar", nd.get("id", "AMBAR-2"), reason)

    # GREEN — every condition must hold.
    green = rules.get("green", {}) or {}
    checks = _green_checks(report, metadata, th)
    if all(passed for _, passed in checks):
        estim = report.get("estimacion") or {}
        reason = _format(green.get("reason_template", ""), {
            "total_eur": estim.get("total_eur"),
            "confidence": estim.get("confidence_overall", 0.0),
        }) or green.get("description", "")
        log.debug("triage decision: verde (%s)", green.get("id", "VERDE-1"))
        return ("verde", green.get("id", "VERDE-1"), reason)

    # AMBER — default; explain which green criteria failed (auditability).
    amber = rules.get("amber", {}) or {}
    missing = ", ".join(name for name, passed in checks if not passed)
    reason = _format(amber.get("reason_template", ""), {"missing_criteria": missing}) \
        or amber.get("description", "")
    log.debug("triage decision: ambar (%s) — faltan: %s", amber.get("id", "AMBAR-1"), missing)
    return ("ambar", amber.get("id", "AMBAR-1"), reason)
