#!/usr/bin/env python3
"""
unify_to_yolo.py — Convierte COCO unificado a formato YOLO segmentación con splits.

Toma el COCO JSON unificado (data/unified/annotations.json), convierte a formato
YOLO segmentación (polígonos normalizados) y crea los splits train/val/test.

Uso:
  python scripts/unify_to_yolo.py
  python scripts/unify_to_yolo.py --stats-only
  python scripts/unify_to_yolo.py --input data/unified --output data/final
"""

import argparse
import json
import logging
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("unify_to_yolo")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "unified"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "final"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"


# =====================================================================
# Conversión COCO → YOLO Segmentación
# =====================================================================

def coco_segmentation_to_yolo(
    segmentation: list,
    img_width: int,
    img_height: int,
    min_points: int = 4,
) -> list[list[float]]:
    """Convierte polígonos COCO a formato YOLO segmentación (normalizado 0-1).
    
    COCO: [[x1, y1, x2, y2, ..., xN, yN], ...]
    YOLO: [[x1, y1, x2, y2, ..., xN, yN], ...]  (normalizado)
    
    Returns:
        Lista de polígonos normalizados. Vacía si no hay polígonos válidos.
    """
    yolo_polygons = []

    # Guard: invalid image dimensions would produce NaN/Inf normalized coords.
    if not img_width or not img_height or img_width <= 0 or img_height <= 0:
        return yolo_polygons

    for polygon in segmentation:
        if not isinstance(polygon, list):
            continue

        # Los polígonos COCO son listas planas [x1,y1,x2,y2,...,xN,yN]
        if len(polygon) < min_points * 2:
            continue

        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            # Clamp a [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.extend([x, y])

        if len(normalized) >= min_points * 2:
            yolo_polygons.append(normalized)

    return yolo_polygons


def convert_coco_to_yolo(
    coco_data: dict,
    output_dir: Path,
    min_bbox_area: float = 100,
    min_polygon_points: int = 4,
) -> dict[str, list]:
    """Convierte todo el dataset COCO a archivos YOLO segmentación.
    
    Returns:
        Dict con image_id → lista de anotaciones YOLO.
    """
    # Indexar imágenes
    images = {img["id"]: img for img in coco_data["images"]}

    # Agrupar anotaciones por imagen
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_labels = {}
    skipped = 0

    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        filename = Path(img_info["file_name"]).stem

        annotations = anns_by_image.get(img_id, [])
        yolo_lines = []

        for ann in annotations:
            class_id = ann["category_id"]
            segmentation = ann.get("segmentation", [])

            # Verificar que no sea RLE
            if isinstance(segmentation, dict):
                skipped += 1
                continue

            # Filtrar por área
            area = ann.get("area", 0)
            if 0 < area < min_bbox_area:
                continue

            # Convertir polígonos
            yolo_polys = coco_segmentation_to_yolo(
                segmentation, img_w, img_h, min_polygon_points
            )

            for poly in yolo_polys:
                coords_str = " ".join(f"{v:.6f}" for v in poly)
                yolo_lines.append(f"{class_id} {coords_str}")

        # Escribir archivo de labels (incluso si vacío, para mantener consistencia)
        label_file = labels_dir / f"{filename}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        image_labels[img_id] = yolo_lines

    if skipped:
        log.warning("  %d anotaciones con RLE omitidas (no soportado)", skipped)

    return image_labels


# =====================================================================
# Splitting estratificado
# =====================================================================

def create_stratified_splits(
    coco_data: dict,
    image_labels: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Crea splits train/val/test estratificados por clase predominante.
    
    Returns:
        Dict con "train", "val", "test" → lista de image_ids.
    """
    random.seed(seed)

    # Determinar clase predominante por imagen
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann["category_id"])

    # Agrupar imágenes por clase predominante
    class_to_images: dict[int, list[int]] = defaultdict(list)
    for img_id, class_ids in anns_by_image.items():
        # Clase predominante = más frecuente
        dominant = Counter(class_ids).most_common(1)[0][0]
        class_to_images[dominant].append(img_id)

    splits = {"train": [], "val": [], "test": []}

    for class_id, img_ids in class_to_images.items():
        random.shuffle(img_ids)
        n = len(img_ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        splits["train"].extend(img_ids[:n_train])
        splits["val"].extend(img_ids[n_train:n_train + n_val])
        splits["test"].extend(img_ids[n_train + n_val:])

    # Shuffle final
    for split in splits.values():
        random.shuffle(split)

    return splits


# =====================================================================
# Leakage-free grouped splitting (perceptual-hash dedup + group-by-vehicle)
# =====================================================================
# WHY: create_stratified_splits shuffles per IMAGE. If the same vehicle/claim
# appears in several photos (or a near-duplicate of an image exists), copies leak
# across train/val/test and inflate the reported metrics. create_grouped_splits
# assigns whole GROUPS (near-duplicate cluster ∪ filename-derived vehicle key) to
# a single split, so no image — and no near-duplicate of it — ever crosses splits.

def _dhash(image_path: Path, hash_size: int = 8) -> "int | None":
    """Difference hash (dHash) of an image as an int. None if unreadable.

    Robust to mild res/compression changes, so near-identical photos collide.
    Lazily imports Pillow/numpy so the module imports without them.
    """
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            small = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
            px = np.asarray(small, dtype=np.int16)
    except Exception:
        return None
    diff = px[:, 1:] > px[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two hashes (popcount of XOR)."""
    return bin(a ^ b).count("1")


class _BKTree:
    """Minimal BK-tree for efficient 'all hashes within distance t' queries."""

    def __init__(self) -> None:
        self._root: "tuple | None" = None  # (hash, key, {dist: child_node})

    def add(self, h: int, key) -> None:
        if self._root is None:
            self._root = (h, key, {})
            return
        node = self._root
        while True:
            d = _hamming(h, node[0])
            child = node[2].get(d)
            if child is None:
                node[2][d] = (h, key, {})
                return
            node = child

    def query(self, h: int, threshold: int) -> list:
        if self._root is None:
            return []
        out: list = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = _hamming(h, node[0])
            if d <= threshold:
                out.append(node[1])
            lo, hi = d - threshold, d + threshold
            for dist, child in node[2].items():
                if lo <= dist <= hi:
                    stack.append(child)
        return out


def _uf_find(parent: dict, x):
    """Union-find with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _uf_union(parent: dict, a, b) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _filename_group_key(stem: str, pattern: "str | None") -> str:
    """Derive a vehicle/claim group key from a filename stem via regex group 1."""
    if pattern:
        m = re.match(pattern, stem)
        if m and m.lastindex:
            return m.group(1) or stem
    return stem


def create_grouped_splits(
    coco_data: dict,
    images_dir: "Path | None",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    group_from_filename: "str | None" = None,
    dedup_phash: bool = True,
    phash_threshold: int = 6,
    class_names: "dict | None" = None,
) -> "tuple[dict[str, list[int]], dict]":
    """Leakage-free splits: cluster near-duplicates + same-vehicle photos into
    groups, then assign whole groups to train/val/test (stratified by each
    group's dominant class).

    Returns:
        (splits, audit) — splits maps "train"/"val"/"test" → image_ids; audit is a
        JSON-serializable dict proving no group spans multiple splits.
    """
    random.seed(seed)
    images = {img["id"]: img for img in coco_data["images"]}

    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann["category_id"])

    # 1) union-find over images, seeded by filename-derived vehicle/claim key
    parent: dict = {img_id: img_id for img_id in images}
    key_rep: dict = {}
    for img_id, img in images.items():
        stem = Path(img["file_name"]).stem
        key = _filename_group_key(stem, group_from_filename)
        if key in key_rep:
            _uf_union(parent, key_rep[key], img_id)
        else:
            key_rep[key] = img_id

    # 2) merge near-duplicate images (perceptual hash) into the same group
    phash_audit = {"computed": 0, "missing": 0, "dup_pairs": 0, "enabled": bool(dedup_phash)}
    if dedup_phash and images_dir is not None:
        tree = _BKTree()
        for img_id, img in images.items():
            h = _dhash(images_dir / img["file_name"])
            if h is None:
                phash_audit["missing"] += 1
                continue
            phash_audit["computed"] += 1
            for nb in tree.query(h, phash_threshold):
                _uf_union(parent, img_id, nb)
                phash_audit["dup_pairs"] += 1
            tree.add(h, img_id)

    # 3) collect groups
    groups: dict = defaultdict(list)
    for img_id in images:
        groups[_uf_find(parent, img_id)].append(img_id)

    def _dominant(img_ids: list) -> int:
        c: Counter = Counter()
        for i in img_ids:
            c.update(anns_by_image.get(i, []))
        return c.most_common(1)[0][0] if c else -1

    # 4) stratify groups by dominant class, assign whole groups to splits
    by_class: dict = defaultdict(list)
    for member_ids in groups.values():
        by_class[_dominant(member_ids)].append(member_ids)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for grouplist in by_class.values():
        random.shuffle(grouplist)
        n = len(grouplist)
        n_train = min(int(round(n * train_ratio)), n)
        n_val = min(int(round(n * val_ratio)), n - n_train)
        for gi, member_ids in enumerate(grouplist):
            if gi < n_train:
                splits["train"].extend(member_ids)
            elif gi < n_train + n_val:
                splits["val"].extend(member_ids)
            else:
                splits["test"].extend(member_ids)
    for ids in splits.values():
        random.shuffle(ids)

    # 5) integrity check: no group may span more than one split
    img_to_split = {img_id: s for s, ids in splits.items() for img_id in ids}
    spanning = 0
    for member_ids in groups.values():
        if len({img_to_split.get(i) for i in member_ids}) > 1:
            spanning += 1

    def _class_counts(ids: list) -> dict:
        c: Counter = Counter()
        for i in ids:
            for cid in anns_by_image.get(i, []):
                c[cid] += 1
        if class_names:
            return {class_names.get(int(k), str(k)): v for k, v in sorted(c.items())}
        return {str(k): v for k, v in sorted(c.items())}

    audit = {
        "strategy": "grouped",
        "seed": seed,
        "n_images": len(images),
        "n_groups": len(groups),
        "phash": phash_audit,
        "ratios_requested": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {
            s: {"images": len(ids), "annotations_by_class": _class_counts(ids)}
            for s, ids in splits.items()
        },
        "cross_split_groups": spanning,  # MUST be 0 — proof of no leakage
    }
    return splits, audit


def organize_splits(
    coco_data: dict,
    splits: dict[str, list[int]],
    source_images_dir: Path,
    source_labels_dir: Path,
    output_dir: Path,
):
    """Organiza archivos en la estructura YOLO: images/{split}/ y labels/{split}/."""
    images = {img["id"]: img for img in coco_data["images"]}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for split_name, img_ids in splits.items():
            img_dir = output_dir / "images" / split_name
            lbl_dir = output_dir / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            task = progress.add_task(
                f"Organizando {split_name}...", total=len(img_ids)
            )

            for img_id in img_ids:
                if img_id not in images:
                    continue

                img_info = images[img_id]
                filename = img_info["file_name"]
                stem = Path(filename).stem

                # Copiar imagen
                src_img = source_images_dir / filename
                if src_img.exists():
                    shutil.copy2(src_img, img_dir / filename)

                # Copiar label
                src_lbl = source_labels_dir / f"{stem}.txt"
                if src_lbl.exists():
                    shutil.copy2(src_lbl, lbl_dir / f"{stem}.txt")

                progress.advance(task)


# =====================================================================
# Estadísticas
# =====================================================================

def compute_statistics(
    coco_data: dict,
    splits: dict[str, list[int]],
    config: dict,
) -> None:
    """Calcula e imprime estadísticas completas del dataset."""
    classes = {int(k): v for k, v in config.get("classes", {}).items()}

    # Anotaciones por imagen
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Tabla principal
    table = Table(title="📊 Distribución del Dataset Final")
    table.add_column("Split", style="bold")
    for cid in sorted(classes):
        table.add_column(classes[cid], justify="right", style="cyan")
    table.add_column("Total imgs", justify="right", style="green")
    table.add_column("Total anns", justify="right", style="yellow")

    grand_total_imgs = 0
    grand_total_anns = 0

    for split_name in ["train", "val", "test"]:
        img_ids = set(splits.get(split_name, []))
        class_counts = Counter()
        total_anns = 0

        for img_id in img_ids:
            for ann in anns_by_image.get(img_id, []):
                class_counts[ann["category_id"]] += 1
                total_anns += 1

        row = [split_name.upper()]
        for cid in sorted(classes):
            row.append(f"{class_counts.get(cid, 0):,}")
        row.append(f"{len(img_ids):,}")
        row.append(f"{total_anns:,}")
        table.add_row(*row)

        grand_total_imgs += len(img_ids)
        grand_total_anns += total_anns

    table.add_section()
    total_class_counts = Counter()
    for ann in coco_data["annotations"]:
        total_class_counts[ann["category_id"]] += 1

    total_row = ["TOTAL"]
    for cid in sorted(classes):
        total_row.append(f"{total_class_counts.get(cid, 0):,}")
    total_row.append(f"{grand_total_imgs:,}")
    total_row.append(f"{grand_total_anns:,}")
    table.add_row(*total_row, style="bold")

    console.print()
    console.print(table)

    # Estadísticas adicionales
    if coco_data["images"]:
        ann_counts = [len(anns_by_image.get(img["id"], [])) for img in coco_data["images"]]
        console.print(f"\n  📐 Anotaciones por imagen: min={min(ann_counts)}, "
                      f"max={max(ann_counts)}, media={sum(ann_counts)/len(ann_counts):.1f}")

    # Balance de clases
    console.print("\n  ⚖️  Balance de clases:")
    total = sum(total_class_counts.values())
    for cid in sorted(classes):
        count = total_class_counts.get(cid, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        console.print(f"     {classes[cid]:15s} {bar} {pct:5.1f}% ({count:,})")


def filter_by_confidence(coco_data: dict, min_conf: float) -> dict:
    """Descarta anotaciones por debajo de `min_conf` (campo `fine_conf` que
    escribe auto_relabel; las anotaciones sin el campo se tratan como conf=1.0).
    Las imágenes que se quedan SIN ninguna anotación se descartan también: no se
    usan como negativo porque contienen daño que no supimos tipar con confianza.
    """
    if min_conf <= 0:
        return coco_data
    anns = coco_data["annotations"]
    kept = [a for a in anns if a.get("fine_conf", 1.0) >= min_conf]
    keep_img_ids = {a["image_id"] for a in kept}
    imgs = [im for im in coco_data["images"] if im["id"] in keep_img_ids]
    log.info("  Filtro conf≥%.2f: anns %d→%d · imágenes %d→%d",
             min_conf, len(anns), len(kept), len(coco_data["images"]), len(imgs))
    out = dict(coco_data)
    out["annotations"] = kept
    out["images"] = imgs
    return out


def generate_dataset_yaml(
    output_dir: Path,
    config: dict,
    yaml_path: Path,
):
    """Genera el archivo dataset.yaml para Ultralytics."""
    classes = config.get("classes", {})
    dataset_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in classes.items()},
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    log.info("Dataset YAML generado: %s", yaml_path)


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte COCO unificado a YOLO segmentación con splits",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Directorio con COCO unificado (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Directorio de salida YOLO (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Archivo de configuración (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Solo mostrar estadísticas sin copiar archivos",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reproducibilidad de splits (default: 42)",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.0,
        help="Piso de confianza (campo fine_conf de auto_relabel). Anns por debajo "
             "se descartan; imágenes que quedan sin anns también. Default 0.0 (sin filtro).",
    )
    parser.add_argument(
        "--dataset-yaml", type=Path, default=None,
        help="Ruta del dataset.yaml a generar (default: configs/dataset.yaml). "
             "Usa otra ruta (p.ej. configs/dataset_v2.yaml) para no pisar la v1.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  COCO → YOLO Segmentación + Splits       [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Cargar config
    config = yaml.safe_load(open(args.config))

    # Cargar COCO unificado
    ann_file = args.input / "annotations.json"
    if not ann_file.exists():
        log.error("No se encuentra %s. Ejecuta primero download_datasets.py", ann_file)
        sys.exit(1)

    with open(ann_file) as f:
        coco_data = json.load(f)

    log.info("Cargado: %d imágenes, %d anotaciones",
             len(coco_data["images"]), len(coco_data["annotations"]))

    # Filtro de confianza (auto_relabel) — quita el ruido de la banda baja
    min_conf = args.min_conf or float(config.get("train_filter", {}).get("min_conf", 0.0))
    coco_data = filter_by_confidence(coco_data, min_conf)

    # Convertir a YOLO
    console.rule("[bold]Conversión COCO → YOLO Segmentación[/]")
    temp_labels_dir = args.output / "labels"
    image_labels = convert_coco_to_yolo(
        coco_data, args.output,
        min_bbox_area=config.get("min_bbox_area", 100),
        min_polygon_points=config.get("min_polygon_points", 4),
    )
    log.info("Convertidas %d imágenes a formato YOLO-seg", len(image_labels))

    # Crear splits
    split_ratios = config.get("splits", {})
    split_cfg = config.get("split", {}) or {}
    strategy = split_cfg.get("strategy", "stratified")
    audit = None

    if strategy == "grouped":
        console.rule("[bold]Splitting Agrupado (leakage-free)[/]")
        classes = {int(k): v for k, v in config.get("classes", {}).items()}
        splits, audit = create_grouped_splits(
            coco_data,
            images_dir=args.input / "images",
            train_ratio=split_ratios.get("train", 0.7),
            val_ratio=split_ratios.get("val", 0.2),
            test_ratio=split_ratios.get("test", 0.1),
            seed=args.seed,
            group_from_filename=split_cfg.get("group_from_filename"),
            dedup_phash=bool(split_cfg.get("dedup_phash", True)),
            phash_threshold=int(split_cfg.get("phash_threshold", 6)),
            class_names=classes,
        )
        log.info("  grupos: %d · imágenes: %d · near-dup pairs: %d · grupos que cruzan splits: %d",
                 audit["n_groups"], audit["n_images"], audit["phash"]["dup_pairs"],
                 audit["cross_split_groups"])
        if audit["cross_split_groups"] != 0:
            log.error("  ⚠ %d grupos cruzan splits — hay fuga. Revisa la lógica de agrupado.",
                      audit["cross_split_groups"])
    else:
        console.rule("[bold]Splitting Estratificado (legacy, por imagen)[/]")
        splits = create_stratified_splits(
            coco_data, image_labels,
            train_ratio=split_ratios.get("train", 0.7),
            val_ratio=split_ratios.get("val", 0.2),
            test_ratio=split_ratios.get("test", 0.1),
            seed=args.seed,
        )

    for split_name, ids in splits.items():
        log.info("  %s: %d imágenes", split_name, len(ids))

    # Auditoría de fuga (siempre que haya estrategia agrupada): prueba escrita de
    # que ningún grupo cruza splits + distribución por clase de cada split.
    if audit is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        audit_path = args.output / "leakage_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        log.info("  Auditoría de fuga: %s", audit_path)

    # Estadísticas
    compute_statistics(coco_data, splits, config)

    if args.stats_only:
        console.print("\n[yellow]--stats-only: no se copiaron archivos.[/]")
        return

    # Organizar en estructura YOLO
    console.rule("[bold]Organizando estructura YOLO[/]")
    source_images = args.input / "images"
    organize_splits(coco_data, splits, source_images, temp_labels_dir, args.output)

    # Limpiar labels temporales de la raíz
    temp_root_labels = args.output / "labels"
    for txt_file in temp_root_labels.glob("*.txt"):
        # Solo borrar los que NO están en subdirectorios
        if txt_file.parent == temp_root_labels:
            txt_file.unlink()

    # Generar dataset.yaml
    yaml_path = args.dataset_yaml or (PROJECT_ROOT / "configs" / "dataset.yaml")
    generate_dataset_yaml(args.output, config, yaml_path)

    console.print(f"\n[bold green]✅ Dataset YOLO listo en: {args.output}[/]")
    console.print(f"   Dataset YAML: {yaml_path}")
    console.print(f"   Siguiente paso: python scripts/train.py\n")


if __name__ == "__main__":
    main()
