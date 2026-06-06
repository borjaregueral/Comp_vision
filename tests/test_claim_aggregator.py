"""
Tests for scripts/claim_aggregator.py (T2.5).

Key case: three photos of the same bumper damage consolidate to ONE damage, not
three. Plus distinct-damage separation, confidence-weighted mean, zone-conflict
voting, conservative max/OR merges, and the empty case.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import claim_aggregator as agg  # noqa: E402


def _det(dtype="scratch", part="front_bumper", zone="front", category="plastic_panel",
         conf=0.9, extension="medium", **extra):
    d = {"type": dtype, "part": part, "zone": zone, "part_category": category,
         "confidence": conf, "extension": extension}
    d.update(extra)
    return d


def _img(h, dets):
    return {"image_hash": h, "damages": dets}


# ── Plan case: 3 views, 1 damage ──────────────────────────────────────

def test_three_views_same_bumper_one_damage():
    reports = [_img("h1", [_det()]), _img("h2", [_det()]), _img("h3", [_det()])]
    out = agg.aggregate_claim(reports)
    assert out["n_raw_detections"] == 3
    assert out["n_consolidated"] == 1
    damage = out["damages"][0]
    assert damage["type"] == "scratch"
    assert damage["part"] == "front_bumper"
    assert damage["supporting_images"] == ["h1", "h2", "h3"]
    assert damage["n_detections"] == 3


def test_two_distinct_damages_stay_separate():
    reports = [
        _img("h1", [_det(dtype="scratch", part="front_bumper")]),
        _img("h2", [_det(dtype="dent", part="front_left_door", zone="front_left",
                         category="body_panel")]),
    ]
    out = agg.aggregate_claim(reports)
    assert out["n_consolidated"] == 2


# ── Consolidation rules ───────────────────────────────────────────────

def test_confidence_weighted_mean():
    reports = [_img("h1", [_det(conf=0.9)]), _img("h2", [_det(conf=0.5)])]
    out = agg.aggregate_claim(reports)
    expected = (0.9 * 0.9 + 0.5 * 0.5) / (0.9 + 0.5)  # Σc²/Σc
    assert out["damages"][0]["confidence"] == pytest.approx(expected, abs=1e-3)


def test_zone_conflict_resolved_by_weighted_vote():
    reports = [
        _img("h1", [_det(zone="front", conf=0.9)]),
        _img("h2", [_det(zone="front", conf=0.8)]),
        _img("h3", [_det(zone="front_left", conf=0.5)]),
    ]
    out = agg.aggregate_claim(reports)
    # front weight 1.7 > front_left 0.5 → front wins.
    assert out["damages"][0]["zone"] == "front"


def test_structural_suspicion_is_or():
    reports = [
        _img("h1", [_det(structural_suspicion=False)]),
        _img("h2", [_det(structural_suspicion=True)]),
    ]
    assert out_struct(reports) is True


def out_struct(reports):
    return agg.aggregate_claim(reports)["damages"][0]["structural_suspicion"]


def test_extension_is_max():
    reports = [_img("h1", [_det(extension="small")]), _img("h2", [_det(extension="large")])]
    assert agg.aggregate_claim(reports)["damages"][0]["extension"] == "large"


def test_severity_is_max_when_present():
    reports = [_img("h1", [_det(severity="leve")]), _img("h2", [_det(severity="severo")])]
    assert agg.aggregate_claim(reports)["damages"][0]["severity"] == "severo"


# ── Grouping fallback & empty ─────────────────────────────────────────

def test_groups_by_zone_when_part_unknown():
    reports = [
        _img("h1", [_det(part="unknown", zone="rear", conf=0.8)]),
        _img("h2", [_det(part="unknown", zone="rear", conf=0.7)]),
    ]
    out = agg.aggregate_claim(reports)
    assert out["n_consolidated"] == 1
    assert out["damages"][0]["zone"] == "rear"


def test_empty_input():
    out = agg.aggregate_claim([])
    assert out["n_consolidated"] == 0
    assert out["damages"] == []
    assert out["n_raw_detections"] == 0
