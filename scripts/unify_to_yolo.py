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
    console.rule("[bold]Splitting Estratificado[/]")
    split_ratios = config.get("splits", {})
    splits = create_stratified_splits(
        coco_data, image_labels,
        train_ratio=split_ratios.get("train", 0.7),
        val_ratio=split_ratios.get("val", 0.2),
        test_ratio=split_ratios.get("test", 0.1),
        seed=args.seed,
    )

    for split_name, ids in splits.items():
        log.info("  %s: %d imágenes", split_name, len(ids))

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
    yaml_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    generate_dataset_yaml(args.output, config, yaml_path)

    console.print(f"\n[bold green]✅ Dataset YOLO listo en: {args.output}[/]")
    console.print(f"   Dataset YAML: {yaml_path}")
    console.print(f"   Siguiente paso: python scripts/train.py\n")


if __name__ == "__main__":
    main()
