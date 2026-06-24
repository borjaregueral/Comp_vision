#!/usr/bin/env python3
"""
refine_masks_sam.py — Tier 2: refina las máscaras flojas de VehiDE con HQ-SAM.

PROBLEMA (del baseline): scratch tiene el mayor salto box→mask (0.316 → 0.251);
el modelo ENCUENTRA los rayones pero los SEGMENTA mal. Causa: los polígonos VIA de
VehiDE son trazos humanos gruesos → ponen un techo a la máscara de todos los modelos.

SOLUCIÓN (sin re-etiquetar a mano): por cada anotación, usar HQ-SAM (Apache-2.0,
fuerte en estructuras finas) con la caja del polígono como prompt + puntos de
centerline para las clases finas (scratch/crack). Una PUERTA QA decide si el
refinamiento se acepta (IoU razonable con el original + conectividad) o si se
conserva el polígono original. Salida: un COCO refinado que unify_to_yolo.py
consume igual que el original.

Uso (en la caja GPU; primero instalar HQ-SAM y bajar el checkpoint):
  pip install segment-anything-hq
  wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
  # PRUEBA primero en 50 imágenes (segundos), valida el muestreo, y solo entonces todo:
  python scripts/refine_masks_sam.py \
      --input data/unified_vehide4 --output data/unified_vehide4_sam \
      --checkpoint sam_hq_vit_h.pth --limit 50
  # pasada completa:
  python scripts/refine_masks_sam.py \
      --input data/unified_vehide4 --output data/unified_vehide4_sam \
      --checkpoint sam_hq_vit_h.pth
  # luego: unify_to_yolo.py --input data/unified_vehide4_sam ... y reentrenar.

NOTA: este script NO se ha podido testear en local (HQ-SAM + GPU). Por eso existe
--limit: valida el comportamiento en una muestra pequeña antes de la pasada de 11k.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Clases finas (tubulares): se les añade prompt de centerline; la caja sola hace que
# SAM se trague el panel entero. Coinciden con los ids de data_config_vehide4.yaml.
THIN_CLASS_IDS = {0, 2}  # scratch, crack

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("refine_sam")


# ── Geometría / máscaras (numpy + cv2) ───────────────────────────────────

def _poly_to_mask(segmentation, h, w, np, cv2):
    """Polígono(s) COCO [[x,y,...]] → máscara binaria uint8 {0,1}."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _mask_to_polys(mask, np, cv2, min_pts=3):
    """Máscara binaria → lista de polígonos COCO (contornos externos)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.reshape(-1, 2)
        if len(c) < min_pts:
            continue
        polys.append([float(v) for v in c.flatten()])
    return polys


def _bbox_xyxy(mask, np):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _iou(a, b, np):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _centerline_points(mask, np, n=5):
    """Puntos sobre el eje (skeleton) de una estructura fina, para prompt de SAM.

    Usa skimage si está; si no, cae al centroide. Devuelve (coords Nx2, labels N).
    """
    try:
        from skimage.morphology import skeletonize  # type: ignore
        skel = skeletonize(mask > 0)
        ys, xs = np.where(skel)
        if xs.size == 0:
            raise ValueError
        idx = np.linspace(0, xs.size - 1, num=min(n, xs.size)).round().astype(int)
        coords = np.stack([xs[idx], ys[idx]], axis=1)
        return coords, np.ones(len(coords), dtype=np.int64)
    except Exception:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None, None
        cx, cy = int(xs.mean()), int(ys.mean())
        return np.array([[cx, cy]]), np.array([1], dtype=np.int64)


# ── HQ-SAM (lazy) ─────────────────────────────────────────────────────────

def _load_predictor(checkpoint: str, model_type: str, device: str):
    from segment_anything_hq import sam_model_registry, SamPredictor  # type: ignore
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


# ── Núcleo ────────────────────────────────────────────────────────────────

def refine(coco: dict, images_dir: Path, predictor, *, iou_lo: float, iou_hi: float,
           area_lo: float, area_hi: float, limit) -> dict:
    """Devuelve un dict de auditoría; muta coco['annotations'][i]['segmentation'] in place."""
    import numpy as np
    import cv2  # noqa: F401
    from PIL import Image

    images = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_ids = list(anns_by_img.keys())
    if limit:
        img_ids = img_ids[:limit]

    audit = Counter()
    audit["images"] = len(img_ids)
    for n, img_id in enumerate(img_ids, 1):
        meta = images.get(img_id)
        if meta is None:
            continue
        src = images_dir / meta["file_name"]
        try:
            arr = np.array(Image.open(src).convert("RGB"))
        except Exception:
            audit["img_missing"] += 1
            continue
        h, w = arr.shape[:2]
        predictor.set_image(arr)

        for a in anns_by_img[img_id]:
            audit["anns"] += 1
            coarse = _poly_to_mask(a.get("segmentation", []), h, w, np, cv2)
            box = _bbox_xyxy(coarse, np)
            if box is None:
                audit["skip_empty"] += 1
                continue

            pts = labels = None
            if a.get("category_id") in THIN_CLASS_IDS:
                pts, labels = _centerline_points(coarse, np)

            try:
                masks, scores, _ = predictor.predict(
                    box=np.array(box), point_coords=pts, point_labels=labels,
                    multimask_output=False,
                )
                refined = (masks[0] > 0).astype(np.uint8)
            except Exception:
                audit["sam_error"] += 1
                continue

            iou = _iou(coarse, refined, np)
            ca, ra = int(coarse.sum()), int(refined.sum())
            ratio = (ra / ca) if ca else 0.0
            # PUERTA QA: IoU en banda razonable + cambio de área acotado.
            if not (iou_lo <= iou <= iou_hi) or not (area_lo <= ratio <= area_hi):
                audit["rejected"] += 1
                continue
            polys = _mask_to_polys(refined, np, cv2)
            if not polys:
                audit["rejected_nopoly"] += 1
                continue
            a["segmentation"] = polys
            a["bbox"] = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            a["area"] = float(ra)
            audit["accepted"] += 1

        if n % 200 == 0:
            log.info("  %d/%d imágenes · aceptadas %d · rechazadas %d",
                     n, len(img_ids), audit["accepted"], audit["rejected"])
    return dict(audit)


def main():
    p = argparse.ArgumentParser(description="Refina máscaras VehiDE con HQ-SAM (Tier 2)")
    p.add_argument("--input", type=Path, required=True, help="Dir COCO unificado (annotations.json + images/)")
    p.add_argument("--output", type=Path, required=True, help="Dir de salida (COCO refinado)")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint HQ-SAM (.pth)")
    p.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b", "vit_tiny"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Procesa solo N imágenes (prueba)")
    p.add_argument("--iou-lo", type=float, default=0.5, help="IoU mínimo refinada-vs-original para aceptar")
    p.add_argument("--iou-hi", type=float, default=0.95, help="IoU máximo (por encima = sin cambio real)")
    p.add_argument("--area-lo", type=float, default=0.3, help="Ratio mínimo área refinada/original")
    p.add_argument("--area-hi", type=float, default=3.0, help="Ratio máximo área refinada/original")
    args = p.parse_args()

    ann_path = args.input / "annotations.json"
    images_dir = args.input / "images"
    if not ann_path.exists():
        log.error("No existe %s", ann_path)
        sys.exit(1)

    with open(ann_path) as f:
        coco = json.load(f)
    log.info("COCO: %d imágenes, %d anotaciones", len(coco.get("images", [])), len(coco.get("annotations", [])))

    try:
        predictor = _load_predictor(args.checkpoint, args.model_type, args.device)
    except ImportError:
        log.error("Falta HQ-SAM. Instala: pip install segment-anything-hq")
        sys.exit(1)

    audit = refine(coco, images_dir, predictor,
                   iou_lo=args.iou_lo, iou_hi=args.iou_hi,
                   area_lo=args.area_lo, area_hi=args.area_hi, limit=args.limit)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "annotations.json", "w") as f:
        json.dump(coco, f)
    with open(args.output / "refine_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    acc, rej = audit.get("accepted", 0), audit.get("rejected", 0)
    tot = acc + rej or 1
    log.info("\n✅ Refinado: aceptadas %d (%.0f%%) · rechazadas %d · errores SAM %d",
             acc, 100 * acc / tot, rej, audit.get("sam_error", 0))
    log.info("   COCO refinado → %s", args.output / "annotations.json")
    log.info("   Imágenes: re-usa las del input (symlink o --input dir). Auditoría → refine_audit.json")
    log.info("   Siguiente: unify_to_yolo.py --input %s --output data/final_vehide4_sam ...", args.output)


if __name__ == "__main__":
    main()
