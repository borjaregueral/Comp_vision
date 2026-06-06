"""
Tests for scripts/audit_log.py (T1.4).

Covers deterministic hashing, that one evaluation produces exactly one parseable
JSONL line, daily rotation, the no-PII guarantee, and the convenience extractor
that builds a record from a validated inference output.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import audit_log as al  # noqa: E402

_H1 = "a" * 64
_H2 = "b" * 64


def _kwargs(**over) -> dict:
    base = dict(
        input_hashes=[_H1],
        model_version={"damage_model": "baseline_v1.0", "parts_model": "parts_seg_v1.0"},
        lane="verde",
        rule_id="VERDE-1",
        n_damages=1,
        processing_time_ms=1234,
        total_eur=300.0,
        id_evaluacion="EVA-20260606-ABCDEF12",
        claim_id="SIN-2026-000123",
        timestamp="2026-06-06T10:00:00Z",
    )
    base.update(over)
    return base


# ── Hashing ───────────────────────────────────────────────────────────

def test_hash_is_deterministic_and_matches_hashlib(tmp_path):
    data = b"fake-image-bytes-\x00\x01\x02"
    f = tmp_path / "img.jpg"
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert al.hash_bytes(data) == expected
    assert al.hash_image(f) == expected
    assert al.hash_image(f) == al.hash_image(f)  # stable across calls


def test_hash_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        al.hash_image(tmp_path / "nope.jpg")


# ── One line per inference ────────────────────────────────────────────

def test_log_inference_writes_one_parseable_line(tmp_path):
    path = al.log_inference(log_dir=tmp_path, **_kwargs())
    assert path.name == "inference_20260606.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["output_summary"] == {"lane": "verde", "total_eur": 300.0, "n_damages": 1}
    assert record["rule_id_applied"] == "VERDE-1"
    assert record["input_hashes"] == [_H1]
    assert record["processing_time_ms"] == 1234


def test_log_inference_appends_same_day(tmp_path):
    al.log_inference(log_dir=tmp_path, **_kwargs())
    path = al.log_inference(log_dir=tmp_path, **_kwargs(id_evaluacion="EVA-20260606-00000002"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(ln) for ln in lines)  # every line parses


def test_daily_rotation_uses_separate_files(tmp_path):
    p1 = al.log_inference(log_dir=tmp_path, **_kwargs(timestamp="2026-01-02T23:59:00Z"))
    p2 = al.log_inference(log_dir=tmp_path, **_kwargs(timestamp="2026-03-15T00:01:00Z"))
    assert p1.name == "inference_20260102.jsonl"
    assert p2.name == "inference_20260315.jsonl"
    assert p1 != p2
    assert len(p1.read_text().splitlines()) == 1
    assert len(p2.read_text().splitlines()) == 1


# ── No PII ────────────────────────────────────────────────────────────

def test_record_keys_are_whitelisted_only():
    record = al.build_record(**_kwargs())
    assert set(record) == al._ALLOWED_KEYS


def test_no_pii_tokens_in_serialized_line(tmp_path):
    # Even if a caller had PII around, build_record has no parameter to carry it.
    path = al.log_inference(log_dir=tmp_path, **_kwargs())
    blob = path.read_text(encoding="utf-8").lower()
    for forbidden in ("matricula", "matrícula", "nombre", "descripcion_asegurado", "apellido"):
        assert forbidden not in blob


# ── Convenience extractor from a validated output ─────────────────────

def test_log_from_output_maps_fields(tmp_path):
    output = {
        "id_evaluacion": "EVA-20260606-DEADBEEF",
        "timestamp": "2026-06-06T12:00:00Z",
        "claim_id": "SIN-2026-000999",
        "model_version": {"damage_model": "baseline_v1.0", "parts_model": "parts_seg_v1.0"},
        "damages": [{"damage_id": "D1"}, {"damage_id": "D2"}],
        "estimacion": {"total_eur": 420.0},
        "lane": "ambar",
        "lane_rule_id": "AMBAR-1",
        "audit": {"input_hashes": [_H1, _H2], "processing_time_ms": 850},
    }
    path = al.log_from_output(output, log_dir=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["id_evaluacion"] == "EVA-20260606-DEADBEEF"
    assert record["rule_id_applied"] == "AMBAR-1"
    assert record["output_summary"] == {"lane": "ambar", "total_eur": 420.0, "n_damages": 2}
    assert record["input_hashes"] == [_H1, _H2]
    assert record["processing_time_ms"] == 850
