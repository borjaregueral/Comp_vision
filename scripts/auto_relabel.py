#!/usr/bin/env python3
"""
auto_relabel.py — Re-etiqueta los daños existentes a la taxonomía fina v2.

NO usa fotos nuevas: re-clasifica los crops que ya tenemos. La etiqueta gruesa
v1 (dent/scratch/crack/broken_light) RESTRINGE las candidatas finas; un
clasificador zero-shot (CLIP) decide solo lo ambiguo (rayón superficial vs
profundo, etc.) y tipa los 20k boxes de Roboflow ("Damage", hoy descartados).
Donde solo hay 1 candidata, la asignación es DIRECTA (sin visión, cero ruido).

Salida: `data/unified_v2/annotations.json` (COCO de 7 clases, con segmentación
intacta + confianza/método por anotación) + symlinks de imágenes + un informe
HTML de revisión por muestreo. Después:

  python scripts/unify_to_yolo.py --input data/unified_v2 \
      --config configs/taxonomy_v2.yaml   # genera data/final v2 + dataset.yaml

Uso:
  python scripts/auto_relabel.py                       # run completo
  python scripts/auto_relabel.py --limit 200 --report  # muestra de validación
  python scripts/auto_relabel.py --no-clip             # solo asignaciones directas
"""

import argparse
import base64
import io
import json
import logging
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("auto_relabel")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "taxonomy_v2.yaml"


# =====================================================================
# Recolección de instancias (de cada fuente COCO → lista uniforme)
# =====================================================================

def collect_instances(cfg: dict) -> tuple[list[dict], dict[str, str]]:
    """Recolecta cada daño localizado como instancia uniforme.

    Returns (instances, image_src). Cada instancia:
      {source, image_path(abs), file_name(out), bbox, segmentation, area,
       coarse, img_w, img_h}
    image_src: file_name_out → abs path origen (para symlinks).
    """
    instances: list[dict] = []
    image_src: dict[str, str] = {}

    for src in cfg["sources"]:
        if src["type"] == "coco":
            ann_files = [(PROJECT_ROOT / src["annotations"], PROJECT_ROOT / src["images_dir"])]
            prefix = src.get("file_prefix", "")
        elif src["type"] == "coco_glob":
            ann_files = [(Path(p), Path(p).parent) for p in sorted(glob(str(PROJECT_ROOT / src["annotations_glob"])))]
            prefix = src.get("file_prefix", "")
        else:
            log.warning("Fuente con type desconocido: %s", src.get("type"))
            continue

        for ann_path, images_dir in ann_files:
            if not ann_path.exists():
                log.warning("  Anotaciones no encontradas: %s (omito)", ann_path)
                continue
            with open(ann_path) as f:
                coco = json.load(f)
            cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
            imgs = {im["id"]: im for im in coco.get("images", [])}
            n_before = len(instances)
            for ann in coco.get("annotations", []):
                im = imgs.get(ann["image_id"])
                if im is None:
                    continue
                bbox = ann.get("bbox") or []
                if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                coarse = cats.get(ann["category_id"], "?")
                file_out = f"{prefix}{im['file_name']}"
                abs_src = str((images_dir / im["file_name"]).resolve())
                image_src[file_out] = abs_src
                instances.append({
                    "source": src["name"],
                    "image_path": abs_src,
                    "file_name": file_out,
                    "bbox": [float(v) for v in bbox],
                    "segmentation": ann.get("segmentation", []),
                    "area": float(ann.get("area", bbox[2] * bbox[3])),
                    "coarse": coarse,
                    "img_w": int(im.get("width", 0)),
                    "img_h": int(im.get("height", 0)),
                })
            log.info("  %s · %s: +%d instancias", src["name"], ann_path.name, len(instances) - n_before)

    return instances, image_src


# =====================================================================
# Recorte
# =====================================================================

def crop_instance(img: Image.Image, bbox: list[float], padding: float) -> Image.Image:
    """Recorta el bbox [x,y,w,h] con margen relativo, clamp a la imagen."""
    x, y, w, h = bbox
    pad = padding * max(w, h)
    left = max(0, int(x - pad))
    top = max(0, int(y - pad))
    right = min(img.width, int(x + w + pad))
    bottom = min(img.height, int(y + h + pad))
    if right <= left or bottom <= top:
        return img.crop((0, 0, min(8, img.width), min(8, img.height)))
    return img.crop((left, top, right, bottom))


# =====================================================================
# Clasificador zero-shot (CLIP vía transformers)
# =====================================================================

class ZeroShotClassifier:
    """CLIP zero-shot restringido a un subconjunto de clases por instancia."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        clf = cfg["classifier"]
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        log.info("  Cargando CLIP %s en %s ...", clf["model_id"], self.device)
        self.model = CLIPModel.from_pretrained(clf["model_id"]).to(self.device).eval()  # type: ignore[arg-type]
        self.processor = CLIPProcessor.from_pretrained(clf["model_id"])
        self.torch = torch

        # Embedding de texto por clase fina (promedio de sus prompts, normalizado)
        self.text_emb: dict = {}
        with torch.no_grad():
            for cname, prompts in cfg["clip_prompts"].items():
                inp = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)  # type: ignore[call-arg]
                t = self._embed(self.model.get_text_features(**inp))  # type: ignore[arg-type]
                t = t / t.norm(dim=-1, keepdim=True)
                v = t.mean(dim=0)
                self.text_emb[cname] = v / v.norm()
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def _embed(self, out):
        """Extrae el embedding en el espacio compartido (512-d) de CLIP.
        transformers>=5 devuelve BaseModelOutputWithPooling cuyo `pooler_output`
        YA es el embedding proyectado; versiones antiguas devuelven el Tensor."""
        torch = self.torch
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output  # type: ignore[union-attr]

    def classify_batch(self, crops: list[Image.Image], candidates: list[list[str]]) -> list[tuple[str, float]]:
        """Para cada crop, softmax sobre SUS candidatas. Returns [(clase, conf)]."""
        torch = self.torch
        with torch.no_grad():
            inp = self.processor(images=crops, return_tensors="pt").to(self.device)  # type: ignore[call-arg]
            feats = self._embed(self.model.get_image_features(**inp))  # type: ignore[arg-type]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out: list[tuple[str, float]] = []
        for i, cand in enumerate(candidates):
            mat = torch.stack([self.text_emb[c] for c in cand])  # [k,d]
            logits = self.logit_scale * (feats[i] @ mat.T)        # [k]
            probs = torch.softmax(logits, dim=-1)
            j = int(torch.argmax(probs).item())
            out.append((cand[j], float(probs[j].item())))
        return out


# =====================================================================
# Asignación de clase fina
# =====================================================================

def assign_fine_labels(
    instances: list[dict],
    cfg: dict,
    use_clip: bool,
    progress: Optional[Progress] = None,
) -> bool:
    """Asigna fine/conf/method/needs_review in-place. Returns clip_ran."""
    coarse_to_fine = cfg["coarse_to_fine"]
    clf = cfg["classifier"]
    pad = float(clf["crop_padding"])
    min_px = int(clf["min_crop_px"])
    thr = float(clf["review_threshold"])
    bs = int(clf["batch_size"])

    queued: list[dict] = []
    for inst in instances:
        cand = coarse_to_fine.get(inst["coarse"])
        if not cand:
            inst.update(fine=None, fine_conf=0.0, fine_method="unmapped", needs_review=True)
            continue
        if len(cand) == 1:
            inst.update(fine=cand[0], fine_conf=1.0, fine_method="direct", needs_review=False)
        else:
            inst["_candidates"] = cand
            queued.append(inst)

    log.info("  Directas: %d · a clasificar (CLIP): %d", len(instances) - len(queued), len(queued))
    if not queued:
        return False

    clip: Optional[ZeroShotClassifier] = None
    if use_clip:
        try:
            clip = ZeroShotClassifier(cfg)
        except Exception as e:  # modelo no descargado / sin red / sin transformers
            log.warning("  [!] No se pudo cargar CLIP (%s).", e)
            log.warning("      Descárgalo donde haya red/GPU y reejecuta. Marco las ambiguas como needs_review.")
            clip = None

    if clip is None:
        for inst in queued:
            inst.update(fine=inst["_candidates"][0], fine_conf=0.0,
                        fine_method="fallback_first", needs_review=True)
            inst.pop("_candidates", None)
        return False

    # Agrupar por imagen para abrir cada archivo una sola vez
    queued.sort(key=lambda x: x["image_path"])
    task = progress.add_task("Clasificando crops (CLIP)...", total=len(queued)) if progress else None

    batch: list[dict] = []
    cur_path: Optional[str] = None
    cur_img: Optional[Image.Image] = None

    def flush(items: list[dict]):
        if not items:
            return
        crops, cands = [], []
        for it in items:
            crops.append(it["_crop"])
            cands.append(it["_candidates"])
        results = clip.classify_batch(crops, cands)  # type: ignore[union-attr]
        for it, (fine, conf) in zip(items, results):
            small = min(it["_crop"].size) < min_px
            it.update(fine=fine, fine_conf=round(conf, 4),
                      fine_method="clip", needs_review=bool(conf < thr or small))
            it.pop("_candidates", None)
            it.pop("_crop", None)
        if progress and task is not None:
            progress.advance(task, len(items))

    for inst in queued:
        if inst["image_path"] != cur_path:
            cur_path = str(inst["image_path"])
            try:
                cur_img = Image.open(cur_path).convert("RGB")
            except Exception:
                cur_img = None
        if cur_img is None:
            inst.update(fine=inst["_candidates"][0], fine_conf=0.0,
                        fine_method="no_image", needs_review=True)
            inst.pop("_candidates", None)
            continue
        inst["_crop"] = crop_instance(cur_img, inst["bbox"], pad)
        batch.append(inst)
        if len(batch) >= bs:
            flush(batch)
            batch = []
    flush(batch)
    return True


# =====================================================================
# Construcción del COCO de salida
# =====================================================================

def build_output_coco(instances: list[dict], cfg: dict) -> dict:
    classes = {int(k): v for k, v in cfg["classes"].items()}
    name_to_id = {v: k for k, v in classes.items()}

    images: dict[str, int] = {}          # file_name → new image id
    coco_images: list[dict] = []
    coco_anns: list[dict] = []

    for inst in instances:
        if inst.get("fine") is None:
            continue
        fn = inst["file_name"]
        if fn not in images:
            images[fn] = len(coco_images)
            coco_images.append({
                "id": images[fn], "file_name": fn,
                "width": inst["img_w"], "height": inst["img_h"],
            })
        coco_anns.append({
            "id": len(coco_anns),
            "image_id": images[fn],
            "category_id": name_to_id[inst["fine"]],
            "segmentation": inst["segmentation"],
            "bbox": inst["bbox"],
            "area": inst["area"],
            "iscrowd": 0,
            # metadatos de auditoría del re-etiquetado
            "fine_conf": inst["fine_conf"],
            "fine_method": inst["fine_method"],
            "needs_review": inst["needs_review"],
            "source": inst["source"],
            "coarse": inst["coarse"],
        })

    return {
        "info": {"description": f"Comp_vision taxonomía {cfg['version']}", "auto_relabel": True},
        "categories": [{"id": cid, "name": name} for cid, name in sorted(classes.items())],
        "images": coco_images,
        "annotations": coco_anns,
    }


def link_images(coco: dict, image_src: dict[str, str], out_dir: Path):
    import os
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in coco["images"]:
        src = image_src.get(im["file_name"])
        if not src or not Path(src).exists():
            continue
        dst = img_dir / im["file_name"]
        try:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
            n += 1
        except OSError as e:
            log.debug("symlink falló (%s); copio", e)
            import shutil
            shutil.copy2(src, dst)
            n += 1
    log.info("  Imágenes enlazadas: %d en %s", n, img_dir)


# =====================================================================
# Informe / estadísticas
# =====================================================================

def print_distribution(coco: dict):
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_class = Counter(a["category_id"] for a in coco["annotations"])
    by_method = Counter(a["fine_method"] for a in coco["annotations"])
    review = sum(1 for a in coco["annotations"] if a["needs_review"])
    total = len(coco["annotations"])

    t = Table(title=f"📊 Distribución v2 ({len(cats)} clases)")
    t.add_column("Clase", style="cyan"); t.add_column("Anns", justify="right")
    t.add_column("%", justify="right"); t.add_column("needs_review", justify="right", style="yellow")
    for cid in sorted(cats):
        n = by_class.get(cid, 0)
        nr = sum(1 for a in coco["annotations"] if a["category_id"] == cid and a["needs_review"])
        t.add_row(cats[cid], f"{n:,}", f"{(100*n/total if total else 0):4.1f}", f"{nr:,}")
    console.print(); console.print(t)
    console.print(f"\n  Método: " + " · ".join(f"{k}={v:,}" for k, v in by_method.most_common()))
    console.print(f"  Total: {total:,} anns · needs_review: {review:,} ({100*review/total if total else 0:.1f}%)")


def write_report(coco: dict, image_src: dict[str, str], out_path: Path, per_class: int):
    """HTML autocontenido: muestra crops por clase con su confianza."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_class: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_class[a["category_id"]].append(a)
    imgs = {im["id"]: im for im in coco["images"]}

    parts = ["<html><head><meta charset='utf-8'><style>",
             "body{font-family:sans-serif;background:#111;color:#eee;padding:16px}",
             "h2{border-bottom:1px solid #444;margin-top:28px}",
             ".grid{display:flex;flex-wrap:wrap;gap:8px}",
             ".cell{width:150px;font-size:11px;text-align:center}",
             ".cell img{width:150px;height:120px;object-fit:cover;border-radius:4px}",
             ".rev{color:#ff8;}.ok{color:#8f8}",
             "</style></head><body>",
             f"<h1>Spot-check re-etiquetado · {coco['info'].get('description','')}</h1>"]

    for cid in sorted(cats):
        items = anns_by_class.get(cid, [])
        # priorizar mostrar mezcla: algunos needs_review y algunos no
        items_sorted = sorted(items, key=lambda a: a["fine_conf"])
        sample = items_sorted[:per_class // 2] + items_sorted[-(per_class - per_class // 2):]
        seen = set(); sample = [a for a in sample if not (a["id"] in seen or seen.add(a["id"]))]
        parts.append(f"<h2>{cats[cid]} <small>({len(items):,} anns)</small></h2><div class='grid'>")
        for a in sample:
            im = imgs[a["image_id"]]
            src = image_src.get(im["file_name"])
            thumb = ""
            if isinstance(src, str) and Path(src).exists():
                try:
                    pil = Image.open(src).convert("RGB")
                    crop = crop_instance(pil, a["bbox"], 0.15)
                    crop.thumbnail((150, 120))
                    buf = io.BytesIO(); crop.save(buf, format="JPEG", quality=70)
                    thumb = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    pass
            cls = "rev" if a["needs_review"] else "ok"
            parts.append(
                f"<div class='cell'><img src='data:image/jpeg;base64,{thumb}'/>"
                f"<div class='{cls}'>{a['fine_method']} {a['fine_conf']:.2f}</div>"
                f"<div>from: {a['coarse']}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")
    log.info("  Informe de revisión: %s", out_path)


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-etiqueta daños a la taxonomía fina v2 (auto-label + revisión)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--limit", type=int, default=None, help="Procesa solo N instancias (validación)")
    p.add_argument("--no-clip", action="store_true", help="Solo asignaciones directas (sin visión)")
    p.add_argument("--report", action="store_true", help="Genera el HTML de spot-check")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Re-etiquetado → Taxonomía v2 (7 clases)  [/]")
    console.print(f"[bold blue]  {cfg['version']}                          [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    console.rule("[bold]Recolección de instancias[/]")
    instances, image_src = collect_instances(cfg)
    log.info("Total instancias localizadas: %d", len(instances))
    if args.limit:
        import random
        random.seed(args.seed)
        random.shuffle(instances)
        instances = instances[:args.limit]
        log.info("  --limit: trabajando con %d", len(instances))

    console.rule("[bold]Asignación de clase fina[/]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        clip_ran = assign_fine_labels(instances, cfg, use_clip=not args.no_clip, progress=prog)

    coco = build_output_coco(instances, cfg)
    print_distribution(coco)

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_out = out_dir / "annotations.json"
    with open(ann_out, "w") as f:
        json.dump(coco, f)
    log.info("COCO v2 escrito: %s (%d imgs, %d anns)", ann_out, len(coco["images"]), len(coco["annotations"]))

    link_images(coco, image_src, out_dir)

    meta = {
        "version": cfg["version"], "clip_ran": clip_ran,
        "n_images": len(coco["images"]), "n_annotations": len(coco["annotations"]),
        "by_class": {c["name"]: sum(1 for a in coco["annotations"] if a["category_id"] == c["id"])
                     for c in coco["categories"]},
        "needs_review": sum(1 for a in coco["annotations"] if a["needs_review"]),
    }
    (out_dir / cfg["output"]["meta_file"]).write_text(json.dumps(meta, indent=2))

    if args.report:
        write_report(coco, image_src, out_dir / cfg["output"]["report_file"],
                     int(cfg["output"]["report_samples_per_class"]))

    console.print(f"\n[bold green]✅ Re-etiquetado listo[/]  (clip_ran={clip_ran})")
    if not clip_ran and not args.no_clip:
        console.print("[yellow]   ⚠ CLIP no corrió: las ambiguas quedaron needs_review. "
                      "Reejecuta donde el modelo CLIP esté disponible.[/]")
    console.print("   Revisa el HTML y luego:")
    console.print(f"   [cyan]python scripts/unify_to_yolo.py --input {cfg['output']['dir']} "
                  f"--config configs/taxonomy_v2.yaml[/]\n")


if __name__ == "__main__":
    main()
