"""
Tests for scripts/severity.py (T2.3).

Covers the plan cases (xenon headlight small crack -> severe; large scratch on a
plastic bumper -> minor/moderate), the max(visual, economic) rule, the bodywork-
crack structural escalation, the tech-light escalation (ESC-3), part->category
resolution, uncatalogued fallback, and the preliminary visual flag used by
predict.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import severity as sv  # noqa: E402


def _damage(part_category=None, part=None, dtype="scratch", extension="small",
            tech=None, structural=False):
    d = {"type": dtype, "extension": extension}
    if part_category is not None:
        d["part_category"] = part_category
    if part is not None:
        d["part"] = part
    if tech is not None:
        d["tech"] = tech
    if structural:
        d["structural_suspicion"] = True
    return d


# ── Plan cases ────────────────────────────────────────────────────────

def test_headlight_small_crack_is_severe():
    out = sv.compute_severity(_damage(part_category="light_assembly", dtype="crack", extension="small"))
    assert out["severity"] == "severo"


def test_large_scratch_plastic_bumper_is_minor_to_moderate():
    out = sv.compute_severity(
        _damage(part_category="plastic_panel", dtype="scratch", extension="large"),
        cost_estimate={"total_eur": 250.0},
    )
    assert out["severity"] in ("leve", "moderado")


# ── max(visual, economic) ─────────────────────────────────────────────

def test_high_cost_raises_severity():
    """plastic scratch small is 'leve' by matrix, but a 1000€ estimate -> severo."""
    out = sv.compute_severity(
        _damage(part_category="plastic_panel", dtype="scratch", extension="small"),
        cost_estimate={"total_eur": 1000.0},
    )
    assert out["matrix_severity"] == "leve"
    assert out["cost_severity"] == "severo"
    assert out["severity"] == "severo"


# ── Escalations ───────────────────────────────────────────────────────

def test_bodywork_crack_sets_structural_and_severe():
    out = sv.compute_severity(_damage(part_category="body_panel", dtype="crack", extension="small"))
    assert out["severity"] == "severo"
    assert out["structural_suspicion"] is True
    assert "ESC-2" in out["escalations"]


def test_tech_headlight_scratch_escalates_esc3():
    out = sv.compute_severity(
        _damage(part_category="light_assembly", dtype="scratch", extension="small", tech="led")
    )
    assert out["severity"] == "severo"
    assert "ESC-3" in out["escalations"]


def test_structural_flag_is_never_leve():
    out = sv.compute_severity(
        _damage(part_category="plastic_panel", dtype="scratch", extension="small", structural=True)
    )
    assert out["structural_suspicion"] is True
    assert out["severity"] != "leve"


# ── Resolution & fallbacks ────────────────────────────────────────────

def test_part_to_category_resolution_from_part():
    # No part_category given; 'front_left_door' -> body_panel -> scratch/small = leve.
    out = sv.compute_severity(_damage(part="front_left_door", dtype="scratch", extension="small"))
    assert out["severity"] == "leve"  # unknown category would have been 'moderado'


def test_uncatalogued_category_defaults_moderado():
    out = sv.compute_severity(_damage(part_category="spoiler", dtype="scratch", extension="small"))
    assert out["severity"] == "moderado"
    assert out["catalogued"] is False


def test_return_shape():
    out = sv.compute_severity(_damage(part_category="mirror", dtype="dent", extension="any"))
    for key in ("severity", "structural_suspicion", "matrix_severity",
                "cost_severity", "catalogued", "escalations"):
        assert key in out


# ── Preliminary visual flag (predict.py) ──────────────────────────────

def test_preliminary_visual_severity_thresholds():
    assert sv.preliminary_visual_severity(1.0) == "Leve"
    assert sv.preliminary_visual_severity(5.0) == "Moderado"
    assert sv.preliminary_visual_severity(15.0) == "Severo"
