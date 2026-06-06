#!/usr/bin/env python3
"""
localize.py — Localización de daños por zona del vehículo.

Combina dos modelos de segmentación:
  1. El modelo de DAÑOS (predict.py): tipo + máscara de cada daño.
  2. Un modelo de PARTES (carparts-seg, 23 clases): segmenta las partes del coche.

Cada daño se asigna a una de 6 zonas (front, rear, front_left, front_right,
rear_left, rear_right) según la parte con la que más solapa su máscara.

Uso:
  python scripts/localize.py --source img.jpg \
      --model runs/damage_seg/phase2_finetune/weights/best.pt \
      --parts-model runs/parts_seg/train/weights/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger("localize")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARTS_CONFIG = PROJECT_ROOT / "configs" / "parts_config.yaml"
DEFAULT_DAMAGE_MODEL = PROJECT_ROOT / "runs" / "damage_seg" / "phase2_finetune" / "weights" / "best.pt"

# Reutiliza la inferencia de daños de predict.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import run_inference, find_images  # noqa: E402


# =====================================================================
# Config
# =====================================================================

def load_parts_config(path: Path = DEFAULT_PARTS_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =====================================================================
# Geometría / máscaras
# =====================================================================

def polygon_to_mask(polygon: list, h: int, w: int) -> np.ndarray:
    """Rasteriza un polígono [[x,y],...] (px abs) a una máscara booleana HxW."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def bbox_to_mask(bbox: list, h: int, w: int) -> np.ndarray:
    """Máscara booleana del rectángulo [x1,y1,x2,y2] (fallback si no hay polígono)."""
    mask = np.zeros((h, w), dtype=bool)
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        mask[y1:y2, x1:x2] = True
    return mask


# =====================================================================
# Asignación de zona (núcleo determinista, testeable)
# =====================================================================

def assign_zone(damage_mask: np.ndarray, parts: list, cfg: dict) -> dict:
    """Asigna una zona a un daño según el solape de su máscara con las partes.

    Args:
        damage_mask: máscara booleana HxW del daño.
        parts: lista de {"class_name": str, "mask": np.ndarray(bool HxW), "conf": float}.
        cfg: parts_config cargado.

    Returns:
        {"zone", "zone_confidence", "matched_part", "side_uncertain", "candidates"}
    """
    part_to_zone = cfg["part_to_zone"]
    side_specific = set(cfg["side_specific_zones"])
    side_agnostic_parts = set(cfg.get("side_agnostic_parts", []))
    a = cfg["assignment"]
    min_overlap = a["min_overlap"]
    w_side = a["side_specific_weight"]
    w_center = a["center_weight"]
    fallback = a.get("fallback_zone", "unknown")

    d_area = int(damage_mask.sum())
    if d_area == 0:
        return {"zone": fallback, "zone_confidence": 0.0, "matched_part": None,
                "side_uncertain": True, "candidates": []}

    candidates = []
    for p in parts:
        zone = part_to_zone.get(p["class_name"])
        if zone is None:
            continue
        inter = int(np.logical_and(damage_mask, p["mask"]).sum())
        overlap = inter / d_area
        if overlap < min_overlap:
            continue
        weight = w_side if zone in side_specific else w_center
        candidates.append({
            "part": p["class_name"],
            "zone": zone,
            "overlap": round(overlap, 4),
            "part_conf": round(float(p.get("conf", 0.0)), 4),
            "score": round(overlap * weight, 4),
        })

    if not candidates:
        return {"zone": fallback, "zone_confidence": 0.0, "matched_part": None,
                "side_uncertain": True, "candidates": []}

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    side_uncertain = (best["zone"] not in side_specific) or (best["part"] in side_agnostic_parts)
    return {
        "zone": best["zone"],
        "zone_confidence": best["overlap"],
        "matched_part": best["part"],
        "side_uncertain": bool(side_uncertain),
        "candidates": candidates,
    }


# =====================================================================
# Inferencia de partes
# =====================================================================

def run_parts_inference(parts_model, image_path: Path, conf: float, imgsz: int) -> list:
    """Devuelve las partes detectadas como [{class_name, mask(bool HxW), conf}]."""
    results = parts_model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return []
    result = results[0]
    names = result.names  # {id: name} del propio modelo de partes
    img_h, img_w = result.orig_shape
    parts = []
    if result.masks is not None and len(result.masks) > 0:
        for i in range(len(result.masks)):
            cls_id = int(result.boxes.cls[i])
            mask = result.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            parts.append({
                "class_name": names.get(cls_id, str(cls_id)),
                "mask": mask_resized,
                "conf": float(result.boxes.conf[i]),
            })
    return parts


def enrich_report_with_zones(report: dict, image_path: Path, parts_model, cfg: dict) -> dict:
    """Añade campos de zona a cada daño del report y un resumen de zonas.

    Mutates and returns `report`. Usa el polígono del daño (o el bbox como
    fallback) para construir la máscara y solaparla con las partes.
    """
    if not report.get("damages"):
        report.setdefault("zones", {})
        return report

    img_w, img_h = report.get("image_size", [None, None])
    if not img_w or not img_h:
        img = cv2.imread(str(image_path))
        if img is None:
            log.warning("No se pudo leer %s para localización", image_path)
            return report
        img_h, img_w = img.shape[:2]

    parts = run_parts_inference(
        parts_model, image_path,
        conf=cfg["assignment"]["parts_conf"],
        imgsz=cfg["assignment"]["parts_imgsz"],
    )

    zone_counts = {z: 0 for z in cfg["zones"]}
    zone_counts[cfg["assignment"].get("fallback_zone", "unknown")] = 0

    for d in report["damages"]:
        poly = d.get("mask_polygon", [])
        dmask = polygon_to_mask(poly, img_h, img_w) if poly else bbox_to_mask(d.get("bbox", []), img_h, img_w)
        z = assign_zone(dmask, parts, cfg)
        d["zone"] = z["zone"]
        d["zone_confidence"] = round(z["zone_confidence"], 4)
        d["matched_part"] = z["matched_part"]
        d["side_uncertain"] = z["side_uncertain"]
        zone_counts[z["zone"]] = zone_counts.get(z["zone"], 0) + 1

    report["zones"] = {z: n for z, n in zone_counts.items() if n > 0}
    report["parts_detected"] = sorted({p["class_name"] for p in parts})
    return report


def print_zone_table(report: dict):
    table = Table(title=f"🚗 Localización — {report.get('image', '')}")
    table.add_column("Daño", style="cyan")
    table.add_column("Tipo")
    table.add_column("Zona", style="green")
    table.add_column("Parte")
    table.add_column("Solape", justify="right")
    table.add_column("Lado", justify="center")
    for i, d in enumerate(report.get("damages", []), 1):
        side = "?" if d.get("side_uncertain") else "✓"
        table.add_row(
            str(i), d.get("class_es", d.get("class", "?")),
            d.get("zone", "unknown"), str(d.get("matched_part") or "-"),
            f"{d.get('zone_confidence', 0):.2f}", side,
        )
    console.print(table)


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Localiza daños por zona del vehículo")
    parser.add_argument("--source", "--image", dest="source", type=Path, required=True,
                        help="Imagen o directorio")
    parser.add_argument("--model", type=str, default=str(DEFAULT_DAMAGE_MODEL),
                        help="Modelo de daños (.pt)")
    parser.add_argument("--parts-model", type=str, default=None,
                        help="Modelo de partes (.pt). Por defecto: el de parts_config.yaml")
    parser.add_argument("--parts-config", type=Path, default=DEFAULT_PARTS_CONFIG)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--no-save-json", dest="save_json", action="store_false")
    return parser.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()
    cfg = load_parts_config(args.parts_config)

    damage_model_path = Path(args.model)
    parts_model_path = Path(args.parts_model or cfg["parts_model"])
    if not parts_model_path.is_absolute():
        parts_model_path = PROJECT_ROOT / parts_model_path

    for label, p in [("daños", damage_model_path), ("partes", parts_model_path)]:
        if not p.exists():
            log.error("Modelo de %s no encontrado: %s", label, p)
            if label == "partes":
                log.error("Entrena el modelo de partes primero: python scripts/train_parts.py")
            sys.exit(1)

    log.info("Cargando modelos...")
    damage_model = YOLO(str(damage_model_path))
    parts_model = YOLO(str(parts_model_path))

    images = find_images(args.source)
    if not images:
        log.error("No se encontraron imágenes en %s", args.source)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for img_path in images:
        report = run_inference(damage_model, img_path, conf=args.conf, imgsz=args.imgsz)
        report = enrich_report_with_zones(report, img_path, parts_model, cfg)
        print_zone_table(report)
        if args.save_json:
            out = args.output / f"{img_path.stem}_localized.json"
            with open(out, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        all_reports.append(report)

    console.print(f"\n[bold green]✅ Localización completada: {len(all_reports)} imagen(es)[/]")


if __name__ == "__main__":
    main()
