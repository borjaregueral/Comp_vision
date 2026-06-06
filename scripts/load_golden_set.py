#!/usr/bin/env python3
"""
load_golden_set.py — Load, validate and stratify the golden set (T3.2).

Reads the per-claim ground-truth JSON files, validates every one against
schemas/ground_truth_v1.json, stratifies by importe tramo (<500 / 500-1500 /
>1500 — the last is a control that should go to the red lane), and reports
distribution statistics (marca, color, provincia, severity, decision, dominant
damage type).

The golden set contains real (PII-bearing) data, so it lives only locally /
encrypted off-repo (golden_set/ is gitignored). This module just loads and
checks it.

Public API
----------
    load_golden_set(golden_dir, *, schema=None, strict=True, tramos=None) -> dict
    stratify(entries, tramos) -> dict
    compute_stats(entries) -> dict
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

log = logging.getLogger("load_golden_set")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
GT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "ground_truth_v1.json"


class GoldenSetValidationError(ValueError):
    """Raised when the golden set contains entries that fail schema validation."""


def _load_gt_schema(path: Optional[Path] = None) -> dict:
    schema_path = Path(path) if path else GT_SCHEMA_PATH
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default_tramos() -> list:
    """Importe tramos, read from business_metrics.yaml (single source of truth)."""
    import yaml

    cfg_path = CONFIG_DIR / "business_metrics.yaml"
    if cfg_path.exists():
        tramos = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("importe_tramos")
        if tramos:
            return tramos
    return [{"name": "<500", "max": 500}, {"name": "500-1500", "max": 1500},
            {"name": ">1500", "max": None}]


# ── Stratification & stats ───────────────────────────────────────────

def stratify(entries: list, tramos: list) -> dict:
    """Assign each entry to the first tramo whose max it falls under (None = catch-all)."""
    strata = {t["name"]: [] for t in tramos}
    for entry in entries:
        importe = entry["importe_final_pagado"]
        for tramo in tramos:
            if tramo["max"] is None or importe < tramo["max"]:
                strata[tramo["name"]].append(entry)
                break
    return strata


def compute_stats(entries: list) -> dict:
    """Distribution statistics over the (valid) golden-set entries."""
    marca, color, prov = Counter(), Counter(), Counter()
    sev, dec, dmg = Counter(), Counter(), Counter()
    importes = []
    for entry in entries:
        vehiculo = entry.get("vehiculo", {}) or {}
        if vehiculo.get("marca"):
            marca[vehiculo["marca"]] += 1
        if vehiculo.get("color_grupo"):
            color[vehiculo["color_grupo"]] += 1
        if vehiculo.get("provincia"):
            prov[vehiculo["provincia"]] += 1
        sev[entry["severidad_oficial"]] += 1
        dec[entry["decision_final"]] += 1
        for tipo in entry.get("tipos_dano", []) or []:
            dmg[tipo] += 1
        importes.append(entry["importe_final_pagado"])

    stats = {
        "n": len(entries),
        "by_marca": dict(marca),
        "by_color_grupo": dict(color),
        "by_provincia": dict(prov),
        "by_severidad": dict(sev),
        "by_decision_final": dict(dec),
        "by_tipo_dano": dict(dmg),
        "tipo_dano_dominante": dmg.most_common(1)[0][0] if dmg else None,
    }
    if importes:
        stats["importe"] = {
            "min": min(importes),
            "max": max(importes),
            "mean": round(sum(importes) / len(importes), 2),
        }
    return stats


# ── Loading ──────────────────────────────────────────────────────────

def load_golden_set(
    golden_dir,
    *,
    schema: Optional[dict] = None,
    strict: bool = True,
    tramos: Optional[list] = None,
) -> dict:
    """Load and validate the golden set.

    Args:
        golden_dir: directory containing per-claim ground-truth *.json files
            (searched recursively).
        schema: pre-loaded ground-truth schema; loaded if None.
        strict: if True, raise GoldenSetValidationError when any entry is invalid.
        tramos: importe tramos; defaults to business_metrics.yaml.

    Returns:
        {entries, n, invalid, strata (tramo→[claim_id]), stats}.

    Raises:
        GoldenSetValidationError: in strict mode, if any entry fails validation.
    """
    from jsonschema import Draft202012Validator

    schema = schema if schema is not None else _load_gt_schema()
    validator = Draft202012Validator(schema)
    tramos = tramos if tramos is not None else _default_tramos()

    golden_dir = Path(golden_dir)
    entries, invalid = [], []
    for path in sorted(golden_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid.append({"path": str(path), "errors": [f"JSON inválido: {exc}"]})
            continue
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            invalid.append({
                "path": str(path),
                "errors": [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                           for e in errors],
            })
        else:
            entries.append(data)

    if strict and invalid:
        detail = "\n".join(f"  {iv['path']}: {iv['errors']}" for iv in invalid)
        raise GoldenSetValidationError(
            f"{len(invalid)} entrada(s) del golden set fallan la validación:\n{detail}"
        )

    strata = stratify(entries, tramos)
    return {
        "entries": entries,
        "n": len(entries),
        "invalid": invalid,
        "strata": {name: [e["claim_id"] for e in members] for name, members in strata.items()},
        "stats": compute_stats(entries),
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Load + validate + stratify the golden set.")
    parser.add_argument("--dir", type=Path, required=True, help="Golden-set directory.")
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help="Do not raise on invalid entries; report them instead.")
    args = parser.parse_args()

    result = load_golden_set(args.dir, strict=args.strict)
    summary = {
        "n": result["n"],
        "n_invalid": len(result["invalid"]),
        "strata": {k: len(v) for k, v in result["strata"].items()},
        "stats": result["stats"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
