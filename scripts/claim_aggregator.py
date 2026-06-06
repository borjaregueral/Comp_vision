#!/usr/bin/env python3
"""
claim_aggregator.py — Multi-view aggregation per claim (T2.5).

A claim has several photos of the same car; the same physical damage shows up in
more than one of them. aggregate_claim merges those repeated detections into one
consolidated damage so cost is computed over UNIQUE damages, not raw detections.

Association: detections are grouped by (type, region), where region is the
specific part when known, else the zone. Detections of the same group coming
from different images are treated as the same physical damage and merged.

Honest limitation (rule 18): a literal "overlap area" across images is not
computable — different viewpoints have no shared pixel space — so we associate by
(type, part/zone), not by cross-image bbox IoU. YOLO's NMS already removes
intra-image duplicates. A rare "two distinct same-type damages on the same part"
is therefore reported as one in v1 (would need cross-view geometric matching).

Consolidation rules:
- confidence: confidence-weighted mean of the members  (Σ c² / Σ c).
- zone / part / part_category: confidence-weighted vote (resolves disagreements).
- extension / severity: the MAX across members (conservative).
- structural_suspicion: OR across members (conservative → feeds the red lane).
- supporting_images: the de-duplicated image hashes backing the damage.

Public API
----------
    aggregate_claim(reports_per_image) -> dict
"""

import logging
from typing import Optional

log = logging.getLogger("claim_aggregator")

_EXT_RANK = {None: -1, "small": 0, "medium": 1, "large": 2}
_SEV_RANK = {"leve": 0, "moderado": 1, "severo": 2}
_UNKNOWNS = {None, "", "unknown"}


def _weighted_vote(pairs):
    """Return the value with the highest total weight, ignoring unknown values."""
    tally = {}
    for value, weight in pairs:
        if value in _UNKNOWNS:
            continue
        tally[value] = tally.get(value, 0.0) + weight
    if not tally:
        return None
    return max(tally, key=tally.get)


def _region_key(det: dict):
    part = det.get("part")
    if part not in _UNKNOWNS:
        return ("part", part)
    zone = det.get("zone")
    if zone not in _UNKNOWNS:
        return ("zone", zone)
    return ("none", "")


def aggregate_claim(reports_per_image: list, config: Optional[dict] = None) -> dict:
    """Consolidate per-image detections into unique damages.

    Args:
        reports_per_image: list of {"image_hash": str, "damages": [detection, ...]}.
            Each detection may carry type, zone, part, part_category, confidence,
            extension, severity, structural_suspicion, area_pct.

    Returns:
        {damages: [consolidated...], n_raw_detections, n_consolidated, supporting_images}
    """
    detections = []
    for report in reports_per_image or []:
        image_hash = report.get("image_hash")
        for det in report.get("damages", []) or []:
            detections.append((image_hash, det))

    groups = {}
    order = []
    for image_hash, det in detections:
        key = (det.get("type"), _region_key(det))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((image_hash, det))

    consolidated = []
    for idx, key in enumerate(order, 1):
        members = groups[key]
        dets = [d for _, d in members]
        confs = [float(d.get("confidence", 0.0)) for d in dets]
        denom = sum(confs)
        if denom > 0:
            conf_weighted = sum(c * c for c in confs) / denom
        else:
            conf_weighted = sum(confs) / len(confs) if confs else 0.0

        def vote(field):
            return _weighted_vote([(d.get(field), float(d.get("confidence", 0.0))) for d in dets])

        extension = max((d.get("extension") for d in dets), key=lambda e: _EXT_RANK.get(e, -1))
        severities = [d.get("severity") for d in dets if d.get("severity") in _SEV_RANK]
        severity = max(severities, key=lambda s: _SEV_RANK[s]) if severities else None
        structural = any(bool(d.get("structural_suspicion")) for d in dets)
        area = max((float(d.get("area_pct", 0.0)) for d in dets), default=0.0)
        support = sorted({h for h, _ in members if h})

        damage = {
            "damage_id": f"C{idx}",
            "type": key[0],
            "zone": vote("zone") or "unknown",
            "part": vote("part") or "unknown",
            "part_category": vote("part_category") or "unknown",
            "extension": extension or "small",
            "confidence": round(conf_weighted, 4),
            "structural_suspicion": structural,
            "supporting_images": support,
            "area_pct": round(area, 2),
            "n_detections": len(dets),
        }
        if severity is not None:
            damage["severity"] = severity
        consolidated.append(damage)

    return {
        "damages": consolidated,
        "n_raw_detections": len(detections),
        "n_consolidated": len(consolidated),
        "supporting_images": sorted({h for h, _ in detections if h}),
    }
