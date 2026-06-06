#!/usr/bin/env python3
"""
download_datasets.py — Descarga y unificación de datasets públicos de daños en vehículos.

Datasets soportados:
  - VehiDE (Kaggle, ~14K imágenes, Apache 2.0)
  - CarDD (HuggingFace mirror, ~4K imágenes)
  - SInfo (Roboflow Universe, ~4.3K imágenes, CC BY 4.0)
  - SYNDCAR (Mendeley, 245 imágenes, CC BY 4.0)

Uso:
  python scripts/download_datasets.py --datasets vehide,cardd,roboflow
  python scripts/download_datasets.py --dry-run
  python scripts/download_datasets.py --skip-download  # Solo procesar datos ya descargados
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

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
log = logging.getLogger("download_datasets")
console = Console()

# ── Directorio del proyecto ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw"
UNIFIED_OUTPUT = PROJECT_ROOT / "data" / "unified"


# =====================================================================
# Utilidades
# =====================================================================

def load_config(config_path: Path) -> dict:
    """Carga la configuración de datos desde YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_class_mapping(config: dict) -> dict[str, Optional[str]]:
    """Devuelve el mapping de clases externas → clases internas."""
    return config.get("class_mapping", {})


def is_excluded_class(class_name: str, config: dict) -> bool:
    """Comprueba si un nombre de clase debe excluirse."""
    patterns = config.get("excluded_patterns", [])
    name_lower = class_name.lower()
    for pattern in patterns:
        if pattern.lower() in name_lower:
            return True
    return False


def map_class_name(class_name: str, config: dict) -> Optional[str]:
    """Mapea un nombre de clase externo a nuestro nombre interno.
    
    Returns:
        Nombre de clase interna, o None si debe excluirse.
    """
    mapping = get_class_mapping(config)

    # Buscar coincidencia exacta
    if class_name in mapping:
        return mapping[class_name]

    # Buscar coincidencia case-insensitive
    for key, value in mapping.items():
        if key.lower() == class_name.lower():
            return value

    # Comprobar patrones de exclusión
    if is_excluded_class(class_name, config):
        return None

    log.warning(f"Clase no mapeada: '{class_name}' — se omitirá")
    return None


def get_internal_class_id(class_name: str, config: dict) -> Optional[int]:
    """Obtiene el ID numérico de una clase interna."""
    classes = config.get("classes", {})
    for cid, cname in classes.items():
        if cname == class_name:
            return int(cid)
    return None


# =====================================================================
# Descargadores por dataset
# =====================================================================

def download_vehide(output_dir: Path, dry_run: bool = False) -> Path:
    """Descarga VehiDE desde Kaggle (~14K imágenes)."""
    dataset_dir = output_dir / "vehide"

    if dry_run:
        log.info("[DRY RUN] Descargaría VehiDE a %s", dataset_dir)
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        console.print("[bold green]⬇ Descargando VehiDE desde Kaggle...[/]")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "hendrichscullen/vehide-dataset",
            path=str(dataset_dir),
            unzip=True,
        )
        log.info("VehiDE descargado en %s", dataset_dir)
    except ImportError:
        log.error(
            "El paquete 'kaggle' no está instalado. "
            "Instálalo con: pip install kaggle\n"
            "Configura tu API key en ~/.kaggle/kaggle.json"
        )
        raise
    except Exception as e:
        log.error("Error descargando VehiDE: %s", e)
        log.info(
            "Descarga manual: https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset\n"
            "Descomprime en: %s", dataset_dir
        )
        raise

    return dataset_dir


def download_cardd(output_dir: Path, dry_run: bool = False) -> Path:
    """Descarga CarDD desde HuggingFace mirror (~4K imágenes)."""
    dataset_dir = output_dir / "cardd"

    if dry_run:
        log.info("[DRY RUN] Descargaría CarDD a %s", dataset_dir)
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        console.print("[bold green]⬇ Descargando CarDD desde HuggingFace...[/]")
        snapshot_download(
            repo_id="harpreetsahota/CarDD",
            repo_type="dataset",
            local_dir=str(dataset_dir),
        )
        log.info("CarDD descargado en %s", dataset_dir)
    except ImportError:
        log.error(
            "El paquete 'huggingface-hub' no está instalado. "
            "Instálalo con: pip install huggingface-hub"
        )
        raise
    except Exception as e:
        log.error("Error descargando CarDD: %s", e)
        log.info(
            "Alternativa: solicitar acceso oficial en https://cardd-ustc.github.io/\n"
            "O descargar manualmente y colocar en: %s", dataset_dir
        )
        raise

    return dataset_dir


def download_roboflow(output_dir: Path, dry_run: bool = False) -> Path:
    """Descarga dataset SInfo de Roboflow Universe (~4.3K imágenes)."""
    dataset_dir = output_dir / "roboflow"

    if dry_run:
        log.info("[DRY RUN] Descargaría SInfo Roboflow a %s", dataset_dir)
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
        console.print("[bold green]⬇ Descargando SInfo desde Roboflow...[/]")

        # API key desde el entorno (no interactivo → no bloquea CI/automatización)
        api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        if not api_key:
            log.warning(
                "ROBOFLOW_API_KEY no definida; se omite la descarga de Roboflow.\n"
                "Para habilitarla: export ROBOFLOW_API_KEY=<tu_key> "
                "(regístrate gratis en https://roboflow.com)."
            )
            return dataset_dir

        rf = Roboflow(api_key=api_key)
        project = rf.workspace("sinfo").project("car-damage-segmentation")
        version = project.version(2)
        # OJO: para proyectos instance-segmentation el formato es
        # "coco-segmentation" (con "coco" la descarga sale vacía).
        version.download("coco-segmentation", location=str(dataset_dir))
        log.info("SInfo Roboflow descargado en %s", dataset_dir)
    except ImportError:
        log.error("El paquete 'roboflow' no está instalado. pip install roboflow")
        raise
    except Exception as e:
        log.error("Error descargando de Roboflow: %s", e)
        raise

    return dataset_dir


def download_syndcar(output_dir: Path, dry_run: bool = False) -> Path:
    """Descarga SYNDCAR desde Mendeley Data (245 imágenes)."""
    dataset_dir = output_dir / "syndcar"

    if dry_run:
        log.info("[DRY RUN] Descargaría SYNDCAR a %s", dataset_dir)
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]⬇ SYNDCAR (Mendeley Data)[/]")
    console.print(
        "[yellow]Descarga manual requerida:[/]\n"
        "  1. Ve a: https://doi.org/10.17632/hzpj48krdt.1\n"
        "  2. Descarga el archivo ZIP\n"
        f"  3. Descomprime en: {dataset_dir}\n"
    )
    log.info("SYNDCAR requiere descarga manual")

    return dataset_dir


DOWNLOADERS = {
    "vehide": download_vehide,
    "cardd": download_cardd,
    "roboflow": download_roboflow,
    "syndcar": download_syndcar,
}


# =====================================================================
# Procesamiento y unificación de anotaciones
# =====================================================================

def find_coco_annotations(dataset_dir: Path) -> list[Path]:
    """Busca archivos de anotaciones COCO JSON en un directorio."""
    patterns = ["*.json", "**/*.json"]
    results = []
    for pattern in patterns:
        for f in dataset_dir.glob(pattern):
            if f.stat().st_size > 1000:  # Ignorar JSONs triviales
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if "annotations" in data and "images" in data:
                        results.append(f)
                except (json.JSONDecodeError, KeyError):
                    continue
    return results


def find_images(dataset_dir: Path, extensions: set[str] = None) -> list[Path]:
    """Busca todas las imágenes en un directorio."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for ext in extensions:
        images.extend(dataset_dir.rglob(f"*{ext}"))
        images.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def _polygon_area(xs: list, ys: list) -> float:
    """Área de un polígono por la fórmula del cordón (shoelace)."""
    n = len(xs)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) / 2.0


def _build_unified(
    images: list,
    annotations: list,
    config: dict,
    dataset_name: str,
) -> dict:
    """Aplica filtros de calidad y arma el COCO unificado.

    Reutiliza la misma lógica de filtrado que ``process_coco_dataset`` para que
    los parsers no-COCO (VIA, FiftyOne) produzcan exactamente el mismo formato.
    Las anotaciones de entrada ya deben tener ``category_id`` interno.
    """
    min_area = config.get("min_bbox_area", 100)
    min_points = config.get("min_polygon_points", 4)

    valid_annotations = []
    filtered_small = 0
    for ann in annotations:
        area = ann.get("area", 0)
        if area > 0 and area < min_area:
            filtered_small += 1
            continue
        seg = ann.get("segmentation", [])
        if seg and isinstance(seg, list) and isinstance(seg[0], list):
            valid_polys = [p for p in seg if len(p) >= min_points * 2]
            if not valid_polys:
                filtered_small += 1
                continue
            ann["segmentation"] = valid_polys
        valid_annotations.append(ann)

    img_ids = {ann["image_id"] for ann in valid_annotations}
    valid_images = [img for img in images if img["id"] in img_ids]

    classes = config.get("classes", {})
    unified_categories = [
        {"id": int(cid), "name": cname, "supercategory": "damage"}
        for cid, cname in classes.items()
    ]

    log.info(
        "  [%s] %d imágenes, %d anotaciones válidas, %d filtradas (pequeñas)",
        dataset_name, len(valid_images), len(valid_annotations), filtered_small,
    )

    return {
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": unified_categories,
        "info": {
            "description": f"Unified damage dataset - source: {dataset_name}",
            "version": "1.0",
        },
    }


def process_vehide_via(dataset_dir: Path, config: dict, dataset_name: str) -> Optional[dict]:
    """Procesa VehiDE en formato VIA (``*_via_annos.json``) → COCO unificado.

    Cada entrada es ``{filename: {name, regions: [{all_x, all_y, class}]}}`` con
    coordenadas en píxeles. Las dimensiones de imagen se leen del archivo (PIL).
    """
    via_files = sorted(dataset_dir.glob("*_via_annos.json")) or sorted(dataset_dir.glob("*via*.json"))
    if not via_files:
        return None

    try:
        from PIL import Image
    except ImportError:
        log.error("Pillow no está instalado (necesario para leer dimensiones de VehiDE).")
        return None

    img_index = {p.name: p for p in find_images(dataset_dir)}
    images: list = []
    annotations: list = []
    img_id = 0
    ann_id = 0
    dims_cache: dict = {}
    missing_imgs = 0

    for vf in via_files:
        log.info("  [vehide] Leyendo VIA: %s", vf.name)
        with open(vf) as fh:
            via = json.load(fh)

        for key, entry in via.items():
            name = Path(entry.get("name", key)).name
            src = img_index.get(name)
            if src is None:
                missing_imgs += 1
                continue

            if src in dims_cache:
                w, h = dims_cache[src]
            else:
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except Exception:
                    missing_imgs += 1
                    continue
                dims_cache[src] = (w, h)

            this_img_id = img_id
            img_id += 1
            has_ann = False

            for region in entry.get("regions", []):
                internal = map_class_name(region.get("class", ""), config)
                if internal is None:
                    continue
                cid = get_internal_class_id(internal, config)
                if cid is None:
                    continue

                xs = region.get("all_x", [])
                ys = region.get("all_y", [])
                if len(xs) < 3 or len(xs) != len(ys):
                    continue

                poly: list = []
                for x, y in zip(xs, ys):
                    poly.append(float(x))
                    poly.append(float(y))

                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                annotations.append({
                    "id": ann_id,
                    "image_id": this_img_id,
                    "category_id": cid,
                    "segmentation": [poly],
                    "bbox": [float(minx), float(miny), float(maxx - minx), float(maxy - miny)],
                    "area": float(_polygon_area(xs, ys)),
                    "iscrowd": 0,
                })
                ann_id += 1
                has_ann = True

            if has_ann:
                images.append({"id": this_img_id, "file_name": name, "width": w, "height": h})

    if missing_imgs:
        log.warning("  [vehide] %d entradas sin imagen en disco (omitidas)", missing_imgs)
    if not images:
        return None
    return _build_unified(images, annotations, config, dataset_name)


def process_cardd_fiftyone(dataset_dir: Path, config: dict, dataset_name: str) -> Optional[dict]:
    """Procesa CarDD en formato FiftyOne (``samples.json``) → COCO unificado.

    Cada detección trae ``bounding_box`` normalizado ``[x,y,w,h]`` y ``mask`` como
    array booleano recortado al bbox, serializado con ``np.save`` + ``zlib`` y
    codificado en base64. El contorno se extrae con OpenCV y se desplaza a
    coordenadas de imagen.
    """
    sample_file = dataset_dir / "samples.json"
    if not sample_file.exists():
        candidates = list(dataset_dir.rglob("samples.json"))
        if not candidates:
            return None
        sample_file = candidates[0]

    import base64
    import io
    import zlib
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.error("opencv-python / numpy no instalados (necesarios para máscaras de CarDD).")
        return None

    with open(sample_file) as fh:
        data = json.load(fh)
    samples = data.get("samples", data) if isinstance(data, dict) else data

    img_index = {p.name: p for p in find_images(dataset_dir)}
    images: list = []
    annotations: list = []
    img_id = 0
    ann_id = 0
    missing_imgs = 0
    mask_failures = 0

    def _decode_mask(mask_field):
        if not (isinstance(mask_field, dict) and "$binary" in mask_field):
            return None
        b64 = mask_field["$binary"].get("base64") if isinstance(mask_field["$binary"], dict) else None
        if not b64:
            return None
        raw = zlib.decompress(base64.b64decode(b64))
        return np.load(io.BytesIO(raw), allow_pickle=False)

    for sample in samples:
        name = Path(sample.get("filepath", "")).name
        src = img_index.get(name)
        if src is None:
            missing_imgs += 1
            continue

        meta = sample.get("metadata", {}) or {}
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            try:
                from PIL import Image
                with Image.open(src) as im:
                    w, h = im.size
            except Exception:
                missing_imgs += 1
                continue

        seg = sample.get("segmentations") or {}
        dets = seg.get("detections", []) if isinstance(seg, dict) else []

        this_img_id = img_id
        img_id += 1
        has_ann = False

        for det in dets:
            internal = map_class_name(det.get("label", ""), config)
            if internal is None:
                continue
            cid = get_internal_class_id(internal, config)
            if cid is None:
                continue

            bb = det.get("bounding_box")
            if not bb or len(bb) != 4:
                continue
            x0, y0, bw, bh = bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h

            poly = None
            try:
                mask = _decode_mask(det.get("mask"))
                if mask is not None:
                    cnts, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if cnts:
                        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
                        if len(c) >= 3:
                            c[:, 0] += x0
                            c[:, 1] += y0
                            poly = c.flatten().tolist()
            except Exception:
                mask_failures += 1

            if poly is None:
                # Fallback: rectángulo del bbox como polígono
                poly = [x0, y0, x0 + bw, y0, x0 + bw, y0 + bh, x0, y0 + bh]

            annotations.append({
                "id": ann_id,
                "image_id": this_img_id,
                "category_id": cid,
                "segmentation": [poly],
                "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1
            has_ann = True

        if has_ann:
            images.append({"id": this_img_id, "file_name": name, "width": w, "height": h})

    if missing_imgs:
        log.warning("  [cardd] %d muestras sin imagen en disco (omitidas)", missing_imgs)
    if mask_failures:
        log.warning("  [cardd] %d máscaras no decodificadas (usado bbox)", mask_failures)
    if not images:
        return None
    return _build_unified(images, annotations, config, dataset_name)


def process_coco_dataset(
    annotation_file: Path,
    images_dir: Path,
    config: dict,
    dataset_name: str,
) -> dict:
    """Procesa un dataset COCO y filtra/mapea clases.
    
    Returns:
        COCO dict unificado con solo las clases objetivo.
    """
    with open(annotation_file) as f:
        coco = json.load(f)

    # Mapear categorías
    original_categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    classes = config.get("classes", {})

    # Crear mapping: old_cat_id → new_cat_id (o None si excluida)
    cat_id_mapping: dict[int, Optional[int]] = {}
    for old_id, old_name in original_categories.items():
        internal_name = map_class_name(old_name, config)
        if internal_name is None:
            cat_id_mapping[old_id] = None
            log.debug("  Clase '%s' (id=%d) → EXCLUIDA", old_name, old_id)
        else:
            new_id = get_internal_class_id(internal_name, config)
            if new_id is not None:
                cat_id_mapping[old_id] = new_id
                log.debug("  Clase '%s' (id=%d) → '%s' (id=%d)", old_name, old_id, internal_name, new_id)
            else:
                cat_id_mapping[old_id] = None
                log.warning("  Clase '%s' mapeada a '%s' pero no tiene ID", old_name, internal_name)

    # Filtrar anotaciones
    min_area = config.get("min_bbox_area", 100)
    min_points = config.get("min_polygon_points", 4)

    valid_annotations = []
    excluded_count = 0
    filtered_small = 0

    for ann in coco.get("annotations", []):
        old_cat_id = ann.get("category_id")
        new_cat_id = cat_id_mapping.get(old_cat_id)

        if new_cat_id is None:
            excluded_count += 1
            continue

        # Filtrar por área mínima
        area = ann.get("area", 0)
        if area > 0 and area < min_area:
            filtered_small += 1
            continue

        # Filtrar por puntos mínimos de polígono
        segmentation = ann.get("segmentation", [])
        if segmentation and isinstance(segmentation, list):
            if isinstance(segmentation[0], list):
                # Polygon format
                valid_polys = [p for p in segmentation if len(p) >= min_points * 2]
                if not valid_polys:
                    filtered_small += 1
                    continue
                ann["segmentation"] = valid_polys

        # Actualizar category_id
        ann["category_id"] = new_cat_id
        valid_annotations.append(ann)

    # Determinar qué imágenes tienen anotaciones válidas
    images_with_annotations = {ann["image_id"] for ann in valid_annotations}
    valid_images = [
        img for img in coco.get("images", [])
        if img["id"] in images_with_annotations
    ]

    # Crear categorías unificadas
    unified_categories = [
        {"id": int(cid), "name": cname, "supercategory": "damage"}
        for cid, cname in classes.items()
    ]

    log.info(
        "  [%s] %d imágenes, %d anotaciones válidas, %d excluidas, %d filtradas (pequeñas)",
        dataset_name, len(valid_images), len(valid_annotations), excluded_count, filtered_small,
    )

    return {
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": unified_categories,
        "info": {
            "description": f"Unified damage dataset - source: {dataset_name}",
            "version": "1.0",
        },
    }


def process_dataset_dir(dataset_dir: Path, config: dict) -> list[dict]:
    """Detecta el formato de un dataset y lo convierte a COCO unificado.

    Soporta: VIA (VehiDE), FiftyOne (CarDD/``samples.json``) y COCO genérico.
    Devuelve una lista de COCO dicts (normalmente uno).
    """
    name = dataset_dir.name

    # 1) VehiDE — formato VIA
    if list(dataset_dir.glob("*_via_annos.json")) or list(dataset_dir.glob("*via*.json")):
        unified = process_vehide_via(dataset_dir, config, name)
        if unified and unified["images"]:
            return [unified]

    # 2) CarDD — formato FiftyOne (samples.json)
    if (dataset_dir / "samples.json").exists() or list(dataset_dir.rglob("samples.json")):
        unified = process_cardd_fiftyone(dataset_dir, config, name)
        if unified and unified["images"]:
            return [unified]

    # 3) COCO genérico (Roboflow / SYNDCAR / otros)
    results: list[dict] = []
    for ann_file in find_coco_annotations(dataset_dir):
        log.info("  Procesando COCO: %s", ann_file.name)
        unified = process_coco_dataset(ann_file, ann_file.parent, config, name)
        if unified["images"]:
            results.append(unified)
    return results


def merge_coco_datasets(datasets: list[dict]) -> dict:
    """Fusiona múltiples COCO dicts en uno solo, re-numerando IDs."""
    merged = {
        "images": [],
        "annotations": [],
        "categories": datasets[0]["categories"] if datasets else [],
        "info": {"description": "Merged vehicle damage dataset", "version": "1.0"},
    }

    image_id_offset = 0
    ann_id_offset = 0

    for ds in datasets:
        # Re-mapear image IDs
        old_to_new_img = {}
        for img in ds["images"]:
            old_id = img["id"]
            new_id = old_id + image_id_offset
            old_to_new_img[old_id] = new_id
            img["id"] = new_id
            merged["images"].append(img)

        # Re-mapear annotation IDs y image_ids
        for ann in ds["annotations"]:
            ann["id"] = ann.get("id", 0) + ann_id_offset
            ann["image_id"] = old_to_new_img.get(ann["image_id"], ann["image_id"])
            merged["annotations"].append(ann)

        if ds["images"]:
            image_id_offset = max(img["id"] for img in merged["images"]) + 1
        if ds["annotations"]:
            ann_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

    return merged


def copy_images_to_unified(
    coco_data: dict,
    source_dirs: list[Path],
    output_dir: Path,
) -> dict:
    """Copia las imágenes referenciadas al directorio unificado y actualiza rutas."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Indexar imágenes disponibles en los directorios fuente
    available: dict[str, Path] = {}
    for src_dir in source_dirs:
        for img_path in find_images(src_dir):
            available[img_path.name] = img_path

    copied = 0
    missing = 0
    updated_images = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Copiando imágenes...", total=len(coco_data["images"]))

        for img_info in coco_data["images"]:
            filename = Path(img_info.get("file_name", "")).name
            src_path = available.get(filename)

            if src_path and src_path.exists():
                dst_path = images_dir / filename
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                img_info["file_name"] = filename
                updated_images.append(img_info)
                copied += 1
            else:
                missing += 1
                log.debug("Imagen no encontrada: %s", filename)

            progress.advance(task)

    coco_data["images"] = updated_images

    # Filtrar anotaciones huérfanas
    valid_img_ids = {img["id"] for img in updated_images}
    coco_data["annotations"] = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in valid_img_ids
    ]

    log.info("Imágenes copiadas: %d, no encontradas: %d", copied, missing)
    return coco_data


def print_statistics(coco_data: dict, config: dict):
    """Imprime estadísticas del dataset unificado."""
    classes = config.get("classes", {})
    class_names = {int(k): v for k, v in classes.items()}

    # Contar instancias por clase
    class_counts: dict[int, int] = {int(k): 0 for k in classes}
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    table = Table(title="📊 Estadísticas del Dataset Unificado")
    table.add_column("Clase", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Instancias", justify="right", style="green")
    table.add_column("% del total", justify="right", style="yellow")

    total = sum(class_counts.values())
    for cid in sorted(class_counts):
        count = class_counts[cid]
        pct = (count / total * 100) if total > 0 else 0
        table.add_row(
            class_names.get(cid, "?"),
            str(cid),
            f"{count:,}",
            f"{pct:.1f}%",
        )

    table.add_section()
    table.add_row("TOTAL", "", f"{total:,}", "100%")

    console.print()
    console.print(table)
    console.print(f"\n  📸 Total imágenes: [bold]{len(coco_data['images']):,}[/]")
    console.print(f"  🏷️  Total anotaciones: [bold]{total:,}[/]")

    if coco_data["images"]:
        avg = total / len(coco_data["images"])
        console.print(f"  📐 Media anotaciones/imagen: [bold]{avg:.1f}[/]")


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y unifica datasets públicos de daños en vehículos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --datasets vehide,cardd
  %(prog)s --dry-run
  %(prog)s --skip-download --datasets vehide
        """,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="vehide,cardd,roboflow,syndcar",
        help="Datasets a descargar (separados por coma). Opciones: vehide,cardd,roboflow,syndcar",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Directorio de salida para datos crudos (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Archivo de configuración (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar lo que se haría, sin descargar nada",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Omitir descarga; solo procesar datos ya presentes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Activar logs de depuración",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Descarga de Datasets — Fotoperitación   [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Cargar config
    if not args.config.exists():
        log.error("Config no encontrada: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    requested = [d.strip().lower() for d in args.datasets.split(",")]

    console.print(f"  Datasets solicitados: [cyan]{', '.join(requested)}[/]")
    console.print(f"  Directorio salida:    [cyan]{args.output}[/]")
    console.print(f"  Dry-run:              [cyan]{args.dry_run}[/]")
    console.print()

    # ── Descargar ─────────────────────────────────────────────────
    dataset_dirs: list[Path] = []
    for name in requested:
        if name not in DOWNLOADERS:
            log.warning("Dataset desconocido: '%s'. Opciones: %s", name, list(DOWNLOADERS.keys()))
            continue

        console.rule(f"[bold]{name.upper()}[/]")
        try:
            if args.skip_download:
                ddir = args.output / name
                if ddir.exists():
                    log.info("Usando datos existentes en %s", ddir)
                    dataset_dirs.append(ddir)
                else:
                    log.warning("Directorio no encontrado: %s", ddir)
            else:
                ddir = DOWNLOADERS[name](args.output, dry_run=args.dry_run)
                dataset_dirs.append(ddir)
        except Exception as e:
            log.error("Fallo en %s: %s", name, e)
            console.print(f"[red]⚠ Continuando con los demás datasets...[/]\n")

    if args.dry_run:
        console.print("\n[yellow]DRY RUN completado. No se descargó nada.[/]")
        return

    # ── Procesar y unificar ──────────────────────────────────────
    console.rule("[bold]Procesamiento y Unificación[/]")

    processed_datasets = []
    for ddir in dataset_dirs:
        log.info("Procesando %s...", ddir.name)

        # Detección automática de formato (VIA / FiftyOne / COCO)
        units = process_dataset_dir(ddir, config)
        if not units:
            log.warning("  No se encontraron anotaciones procesables en %s", ddir)
            continue

        for unified in units:
            processed_datasets.append((unified, ddir))

    if not processed_datasets:
        console.print("[red]No se encontraron datos procesables. Verifica las descargas.[/]")
        sys.exit(1)

    # Fusionar todos los datasets
    console.print("\n[bold]Fusionando datasets...[/]")
    merged = merge_coco_datasets([ds for ds, _ in processed_datasets])

    # Copiar imágenes al directorio unificado
    unified_dir = UNIFIED_OUTPUT
    unified_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [ddir for _, ddir in processed_datasets]
    merged = copy_images_to_unified(merged, source_dirs, unified_dir)

    # Guardar anotaciones unificadas
    ann_output = unified_dir / "annotations.json"
    with open(ann_output, "w") as f:
        json.dump(merged, f, indent=2)
    log.info("Anotaciones unificadas guardadas en %s", ann_output)

    # ── Estadísticas ─────────────────────────────────────────────
    print_statistics(merged, config)

    console.print(f"\n[bold green]✅ Dataset unificado listo en: {unified_dir}[/]")
    console.print(f"   Siguiente paso: python scripts/unify_to_yolo.py\n")


if __name__ == "__main__":
    main()
