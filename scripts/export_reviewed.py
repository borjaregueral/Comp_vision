#!/usr/bin/env python3
"""
export_reviewed.py — Importa anotaciones revisadas desde CVAT y las fusiona con el dataset.

Lee la exportación de CVAT (COCO JSON o CVAT XML) y la integra en el dataset
final en formato YOLO segmentación.

Uso:
  python scripts/export_reviewed.py --input path/to/cvat_export.zip --dataset data/final
  python scripts/export_reviewed.py --input path/to/annotations.json --merge-strategy append
"""

import argparse
import json
import logging
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("export_reviewed")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"

CLASS_IDS = {"dent": 0, "scratch": 1, "crack": 2, "broken_light": 3}


# =====================================================================
# Parsers de exportación
# =====================================================================

def parse_cvat_xml(xml_path: Path) -> tuple[list[dict], list[dict]]:
    """Parsea exportación CVAT XML a formato COCO-like."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images = []
    annotations = []
    ann_id = 1

    for img_el in root.findall("image"):
        img_id = int(img_el.get("id", 0))
        filename = img_el.get("name", "")
        width = int(img_el.get("width", 0))
        height = int(img_el.get("height", 0))

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for poly_el in img_el.findall("polygon"):
            label = poly_el.get("label", "")
            class_id = CLASS_IDS.get(label)
            if class_id is None:
                log.warning("Clase desconocida en CVAT: '%s'", label)
                continue

            points_str = poly_el.get("points", "")
            if not points_str:
                continue

            # CVAT format: "x1,y1;x2,y2;..."
            polygon = []
            for point in points_str.split(";"):
                parts = point.strip().split(",")
                if len(parts) == 2:
                    polygon.extend([float(parts[0]), float(parts[1])])

            if len(polygon) >= 8:  # Mín 4 puntos
                # Calcular bbox
                xs = polygon[0::2]
                ys = polygon[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": [polygon],
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                })
                ann_id += 1

    return images, annotations


def parse_coco_json(json_path: Path) -> tuple[list[dict], list[dict]]:
    """Parsea exportación COCO JSON."""
    with open(json_path) as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # Remapear categorías si es necesario
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    for ann in annotations:
        cat_name = categories.get(ann["category_id"], "")
        new_id = CLASS_IDS.get(cat_name)
        if new_id is not None:
            ann["category_id"] = new_id
        else:
            # Intentar si el ID ya es correcto
            if ann["category_id"] not in CLASS_IDS.values():
                log.warning("Categoría no mapeada: id=%d name='%s'",
                           ann["category_id"], cat_name)

    return images, annotations


def parse_export(input_path: Path) -> tuple[list[dict], list[dict], Path]:
    """Detecta formato y parsea la exportación.
    
    Returns:
        (images, annotations, images_dir)
    """
    images_dir = None

    if input_path.suffix == ".zip":
        # Descomprimir ZIP
        extract_dir = input_path.parent / "cvat_extracted"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(extract_dir)

        # Buscar anotaciones dentro del ZIP
        xml_files = list(extract_dir.rglob("*.xml"))
        json_files = list(extract_dir.rglob("*.json"))

        if xml_files:
            images, annotations = parse_cvat_xml(xml_files[0])
        elif json_files:
            images, annotations = parse_coco_json(json_files[0])
        else:
            log.error("No se encontraron anotaciones en el ZIP")
            sys.exit(1)

        # Buscar directorio de imágenes
        data_dir = extract_dir / "data"
        if data_dir.exists():
            images_dir = data_dir
        else:
            images_dir = extract_dir

    elif input_path.suffix == ".xml":
        images, annotations = parse_cvat_xml(input_path)
        images_dir = input_path.parent

    elif input_path.suffix == ".json":
        images, annotations = parse_coco_json(input_path)
        images_dir = input_path.parent

    else:
        log.error("Formato no soportado: %s", input_path.suffix)
        sys.exit(1)

    return images, annotations, images_dir


# =====================================================================
# Conversión a YOLO y merge
# =====================================================================

def annotations_to_yolo(
    annotations: list[dict],
    images: list[dict],
) -> dict[str, list[str]]:
    """Convierte anotaciones a formato YOLO segmentación.
    
    Returns:
        Dict filename_stem → lista de líneas YOLO.
    """
    img_info = {img["id"]: img for img in images}

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    yolo_data = {}
    for img_id, img in img_info.items():
        w, h = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        lines = []

        for ann in anns_by_image.get(img_id, []):
            segmentation = ann.get("segmentation", [])
            if not segmentation or not isinstance(segmentation[0], list):
                continue

            for polygon in segmentation:
                if len(polygon) < 8:
                    continue

                # Normalizar
                normalized = []
                for i in range(0, len(polygon), 2):
                    nx = max(0.0, min(1.0, polygon[i] / w))
                    ny = max(0.0, min(1.0, polygon[i + 1] / h))
                    normalized.extend([nx, ny])

                coords = " ".join(f"{v:.6f}" for v in normalized)
                lines.append(f"{ann['category_id']} {coords}")

        yolo_data[stem] = lines

    return yolo_data


def merge_into_dataset(
    yolo_data: dict[str, list[str]],
    images: list[dict],
    images_dir: Path,
    dataset_dir: Path,
    strategy: str = "update",
):
    """Fusiona las anotaciones revisadas en el dataset existente.
    
    Args:
        strategy: 'update' = reemplazar si existe, 'append' = añadir como nuevas.
    """
    # Determinar a qué split van las nuevas imágenes
    # Si ya existen en un split, mantener ahí; si no, asignar a train
    existing_images = {}
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        if img_dir.exists():
            for img_path in img_dir.iterdir():
                existing_images[img_path.stem] = split

    added = 0
    updated = 0

    for img_info in images:
        stem = Path(img_info["file_name"]).stem
        filename = img_info["file_name"]

        if stem not in yolo_data:
            continue

        # Determinar split
        if stem in existing_images:
            split = existing_images[stem]
            if strategy == "update":
                updated += 1
            else:
                continue  # Skip en modo append si ya existe
        else:
            split = "train"  # Nuevas van a train
            added += 1

        # Escribir label
        label_dir = dataset_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_file = label_dir / f"{stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_data[stem]))

        # Copiar imagen si no existe
        img_dst_dir = dataset_dir / "images" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        img_dst = img_dst_dir / filename

        if not img_dst.exists() and images_dir:
            # Buscar en el directorio de imágenes
            src_candidates = [
                images_dir / filename,
                images_dir / "data" / filename,
            ]
            for src in src_candidates:
                if src.exists():
                    shutil.copy2(src, img_dst)
                    break

    return added, updated


def count_dataset_stats(dataset_dir: Path) -> dict:
    """Cuenta estadísticas del dataset."""
    stats = {}
    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            stats[split] = {"images": 0, "annotations": 0, "classes": Counter()}
            continue

        n_images = 0
        n_anns = 0
        class_counts = Counter()

        for label_file in labels_dir.glob("*.txt"):
            n_images += 1
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        n_anns += 1
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

        stats[split] = {"images": n_images, "annotations": n_anns, "classes": class_counts}

    return stats


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Importa anotaciones revisadas de CVAT al dataset",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Archivo de exportación CVAT (ZIP, XML o COCO JSON)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "final",
        help="Directorio del dataset YOLO existente",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--merge-strategy", choices=["update", "append"], default="update",
        help="update: reemplaza si existe, append: solo añade nuevas (default: update)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Importar Anotaciones Revisadas (CVAT)    [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    if not args.input.exists():
        log.error("Archivo no encontrado: %s", args.input)
        sys.exit(1)

    # Estadísticas ANTES
    console.rule("[bold]Estado ANTES[/]")
    stats_before = count_dataset_stats(args.dataset)
    for split, s in stats_before.items():
        console.print(f"  {split:5s}: {s['images']:5d} imgs, {s['annotations']:5d} anns")

    # Parsear exportación
    console.rule("[bold]Parseando exportación[/]")
    images, annotations, images_dir = parse_export(args.input)
    log.info("Parseadas: %d imágenes, %d anotaciones", len(images), len(annotations))

    # Convertir a YOLO
    yolo_data = annotations_to_yolo(annotations, images)

    # Merge
    console.rule("[bold]Fusionando con dataset[/]")
    added, updated = merge_into_dataset(
        yolo_data, images, images_dir,
        args.dataset, args.merge_strategy,
    )

    # Estadísticas DESPUÉS
    console.rule("[bold]Estado DESPUÉS[/]")
    stats_after = count_dataset_stats(args.dataset)

    # Tabla comparativa
    class_names = {0: "dent", 1: "scratch", 2: "crack", 3: "broken_light"}
    table = Table(title="📊 Antes vs Después")
    table.add_column("Split")
    table.add_column("Imgs (antes)", justify="right")
    table.add_column("Imgs (después)", justify="right", style="green")
    table.add_column("Anns (antes)", justify="right")
    table.add_column("Anns (después)", justify="right", style="green")

    for split in ["train", "val", "test"]:
        b = stats_before.get(split, {"images": 0, "annotations": 0})
        a = stats_after.get(split, {"images": 0, "annotations": 0})
        table.add_row(
            split,
            str(b["images"]), str(a["images"]),
            str(b["annotations"]), str(a["annotations"]),
        )

    console.print()
    console.print(table)
    console.print(f"\n  ➕ Añadidas: {added} imágenes nuevas")
    console.print(f"  🔄 Actualizadas: {updated} imágenes existentes")

    # Regenerar dataset.yaml
    config = yaml.safe_load(open(args.config))
    yaml_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    dataset_config = {
        "path": str(args.dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in config.get("classes", {}).items()},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[bold green]✅ Dataset actualizado[/]")
    console.print(f"   Dataset YAML regenerado: {yaml_path}")
    console.print(f"   Siguiente paso: python scripts/train.py\n")


if __name__ == "__main__":
    main()
