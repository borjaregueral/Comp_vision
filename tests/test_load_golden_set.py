"""
Tests for scripts/load_golden_set.py (T3.2).

Validates loading against the ground-truth schema (rejects missing fields and
PII), correct stratification by importe tramo (incl. boundaries), and the
distribution statistics. Uses a synthetic golden set written to tmp_path.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import load_golden_set as lgs  # noqa: E402


def _gt(claim, importe, *, marca="Seat", color="claro", prov="Zaragoza",
        sev="moderado", struct=False, dec="resuelto_sin_peritaje", tipos=("scratch",)):
    return {
        "claim_id": claim, "importe_final_pagado": importe, "moneda": "EUR",
        "severidad_oficial": sev, "es_estructural": struct, "decision_final": dec,
        "vehiculo": {"marca": marca, "modelo": "X", "anio": 2019,
                     "color_grupo": color, "provincia": prov},
        "tipos_dano": list(tipos),
    }


def _write(dir_, gt):
    (dir_ / f"{gt['claim_id']}.json").write_text(json.dumps(gt), encoding="utf-8")


# ── Loading & validation ──────────────────────────────────────────────

def test_loads_valid_entries(tmp_path):
    for i, imp in enumerate([300, 800, 1600]):
        _write(tmp_path, _gt(f"C{i}", imp))
    result = lgs.load_golden_set(tmp_path)
    assert result["n"] == 3
    assert result["invalid"] == []


def test_rejects_missing_required_field_strict(tmp_path):
    _write(tmp_path, _gt("OK", 300))
    (tmp_path / "BAD.json").write_text(json.dumps({"claim_id": "BAD"}), encoding="utf-8")
    with pytest.raises(lgs.GoldenSetValidationError):
        lgs.load_golden_set(tmp_path)


def test_invalid_collected_when_not_strict(tmp_path):
    _write(tmp_path, _gt("OK", 300))
    (tmp_path / "BAD.json").write_text(json.dumps({"claim_id": "BAD"}), encoding="utf-8")
    result = lgs.load_golden_set(tmp_path, strict=False)
    assert result["n"] == 1
    assert len(result["invalid"]) == 1
    assert "BAD.json" in result["invalid"][0]["path"]


def test_pii_entry_rejected(tmp_path):
    gt = _gt("PII", 300)
    gt["matricula"] = "1234ABC"  # additionalProperties:false → invalid
    _write(tmp_path, gt)
    with pytest.raises(lgs.GoldenSetValidationError):
        lgs.load_golden_set(tmp_path)


def test_empty_dir(tmp_path):
    result = lgs.load_golden_set(tmp_path)
    assert result["n"] == 0
    assert all(len(v) == 0 for v in result["strata"].values())


def test_malformed_json_is_reported(tmp_path):
    _write(tmp_path, _gt("OK", 300))
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    result = lgs.load_golden_set(tmp_path, strict=False)
    assert result["n"] == 1
    assert any("broken.json" in iv["path"] and "JSON inválido" in iv["errors"][0]
               for iv in result["invalid"])


# ── Stratification ────────────────────────────────────────────────────

def test_stratify_by_tramo(tmp_path):
    _write(tmp_path, _gt("A", 300))    # <500
    _write(tmp_path, _gt("B", 800))    # 500-1500
    _write(tmp_path, _gt("C", 1600))   # >1500
    strata = lgs.load_golden_set(tmp_path)["strata"]
    assert strata["<500"] == ["A"]
    assert strata["500-1500"] == ["B"]
    assert strata[">1500"] == ["C"]


def test_stratify_boundaries(tmp_path):
    _write(tmp_path, _gt("E500", 500))     # 500 → 500-1500 (lower bound inclusive)
    _write(tmp_path, _gt("E1500", 1500))   # 1500 → >1500
    strata = lgs.load_golden_set(tmp_path)["strata"]
    assert "E500" in strata["500-1500"]
    assert "E1500" in strata[">1500"]


# ── Stats ─────────────────────────────────────────────────────────────

def test_stats_distributions(tmp_path):
    _write(tmp_path, _gt("A", 300, marca="Seat", color="claro", tipos=("scratch",)))
    _write(tmp_path, _gt("B", 400, marca="Seat", color="oscuro", tipos=("scratch", "dent")))
    _write(tmp_path, _gt("C", 350, marca="Renault", color="claro", tipos=("dent",)))
    stats = lgs.load_golden_set(tmp_path)["stats"]
    assert stats["by_marca"] == {"Seat": 2, "Renault": 1}
    assert stats["by_color_grupo"] == {"claro": 2, "oscuro": 1}
    assert stats["tipo_dano_dominante"] in {"scratch", "dent"}  # scratch=3, dent=2 → scratch
    assert stats["tipo_dano_dominante"] == "scratch"
    assert stats["importe"]["min"] == 300
