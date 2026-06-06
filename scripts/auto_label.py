#!/usr/bin/env python3
"""
auto_label.py — Auto-etiquetado con Autodistill (GroundingDINO + SAM).

Genera máscaras de segmentación automáticas para imágenes sin anotar
usando modelos foundation (zero-shot).

Uso:
  python scripts/auto_label.py --input data/raw/unlabeled --output data/auto_labeled
  python scripts/auto_label.py --input data/raw/unlabeled --limit 10 --visualize
  python scripts/auto_label.py --device cpu
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
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
log = logging.getLogger("auto_label")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "auto_label_config.yaml"

# Colores por clase (BGR para OpenCV)
CLASS_COLORS = {
    "dent": (68, 68, 255),          # Rojo
    "scratch": (0, 215, 255),       # Amarillo
    "crack": (255, 136, 68),        # Azul
    "broken_light": (255, 68, 170), # Morado
}

CLASS_IDS = {"dent": 0, "scratch": 1, "crack": 2, "broken_light": 3}


# =====================================================================
# Auto-labeling con Autodistill
# =====================================================================

def build_ontology(config: dict):
    """Construye la ontología de Autodistill desde el config."""
    try:
        from autodistill.detection import CaptionOntology
    except ImportError:
        log.error(
            "autodistill no está instalado.\n"
            "Instala con: pip install autodistill autodistill-grounded-sam"
        )
        sys.exit(1)

    ontology_config = config.get("ontology", {})
    mapping = {}
    for class_name, class_cfg in ontology_config.items():
        prompts = class_cfg.get("prompts", class_name)
        mapping[prompts] = class_name

    return CaptionOntology(mapping)


def init_model(config: dict, device: str = "auto"):
    """Inicializa el modelo GroundedSAM de Autodistill."""
    try:
        from autodistill_grounded_sam import GroundedSAM
    except ImportError:
        log.error(
            "autodistill-grounded-sam no está instalado.\n"
            "Instala con: pip install autodistill-grounded-sam"
        )
        sys.exit(1)

    ontology = build_ontology(config)

    # Thresholds de GroundingDINO desde el config (antes se ignoraban → defaults)
    gd_config = config.get("grounding_dino", {})
    box_threshold = gd_config.get("box_threshold", 0.35)
    text_threshold = gd_config.get("text_threshold", 0.25)

    console.print("[bold]Inicializando GroundedSAM...[/]")
    console.print(f"  box_threshold={box_threshold}, text_threshold={text_threshold}")
    console.print("  (Primera ejecución descargará los modelos, ~2-5 min)")

    model = GroundedSAM(
        ontology=ontology,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    log.info("Modelo inicializado correctamente")
    return model


def find_images(input_dir: Path) -> list[Path]:
    """Encuentra todas las imágenes en un directorio."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_auto_labeling(
    model,
    input_dir: Path,
    output_dir: Path,
    config: dict,
    limit: int = 0,
    visualize: bool = True,
):
    """Ejecuta el auto-etiquetado en batch.
    
    Args:
        model: Modelo GroundedSAM inicializado.
        input_dir: Directorio con imágenes sin anotar.
        output_dir: Directorio de salida.
        config: Configuración del auto-labeling.
        limit: Límite de imágenes a procesar (0 = todas).
        visualize: Si True, genera visualizaciones con masks.
    """
    import supervision as sv

    images = find_images(input_dir)
    if not images:
        log.error("No se encontraron imágenes en %s", input_dir)
        return

    if limit > 0:
        images = images[:limit]
        log.info("Limitado a %d imágenes", limit)

    # Crear directorios de salida
    labels_dir = output_dir / "labels"
    images_out_dir = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = output_dir / "visualizations"
    if visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Obtener thresholds por clase
    ontology_config = config.get("ontology", {})
    class_names = list(ontology_config.keys())

    # Estadísticas
    confidence_scores = {}
    total_detections = {name: 0 for name in CLASS_IDS}
    processed = 0
    errors = 0

    # COCO format acumulativo
    coco_images = []
    coco_annotations = []
    ann_id = 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Auto-etiquetando...", total=len(images))

        for img_path in images:
            try:
                # Leer imagen
                image = cv2.imread(str(img_path))
                if image is None:
                    log.warning("No se pudo leer: %s", img_path.name)
                    errors += 1
                    progress.advance(task)
                    continue

                h, w = image.shape[:2]

                # Ejecutar predicción
                detections = model.predict(str(img_path))

                # Procesar detecciones
                yolo_lines = []
                img_confidences = []

                if detections and len(detections) > 0:
                    for i in range(len(detections)):
                        class_idx = detections.class_id[i]
                        confidence = detections.confidence[i] if detections.confidence is not None else 1.0

                        if class_idx >= len(class_names):
                            continue

                        class_name = class_names[class_idx]
                        internal_id = CLASS_IDS.get(class_name)
                        if internal_id is None:
                            continue

                        # Filtrar por threshold de confianza
                        threshold = ontology_config.get(class_name, {}).get(
                            "confidence_threshold", 0.35
                        )
                        if confidence < threshold:
                            continue

                        # Extraer máscara si disponible
                        if detections.mask is not None and detections.mask[i] is not None:
                            mask = detections.mask[i].astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )

                            for contour in contours:
                                if len(contour) < 4:
                                    continue

                                # Convertir contorno a polígono YOLO normalizado
                                polygon = contour.flatten().tolist()
                                normalized = []
                                for j in range(0, len(polygon), 2):
                                    nx = polygon[j] / w
                                    ny = polygon[j + 1] / h
                                    normalized.extend([
                                        max(0.0, min(1.0, nx)),
                                        max(0.0, min(1.0, ny)),
                                    ])

                                if len(normalized) >= 8:  # Mín 4 puntos
                                    coords_str = " ".join(f"{v:.6f}" for v in normalized)
                                    yolo_lines.append(f"{internal_id} {coords_str}")

                                    # COCO annotation
                                    coco_ann = {
                                        "id": ann_id,
                                        "image_id": processed,
                                        "category_id": internal_id,
                                        "segmentation": [polygon],
                                        "bbox": list(cv2.boundingRect(contour)),
                                        "area": float(cv2.contourArea(contour)),
                                        "iscrowd": 0,
                                        "confidence": float(confidence),
                                    }
                                    coco_annotations.append(coco_ann)
                                    ann_id += 1

                        else:
                            # Solo bounding box (sin máscara)
                            bbox = detections.xyxy[i]
                            x1, y1, x2, y2 = bbox
                            # Crear polígono rectangular
                            normalized = [
                                x1/w, y1/h, x2/w, y1/h,
                                x2/w, y2/h, x1/w, y2/h,
                            ]
                            coords_str = " ".join(f"{v:.6f}" for v in normalized)
                            yolo_lines.append(f"{internal_id} {coords_str}")

                        total_detections[class_name] += 1
                        img_confidences.append(float(confidence))

                # Guardar labels YOLO
                label_file = labels_dir / f"{img_path.stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

                # Copiar imagen
                dst_img = images_out_dir / img_path.name
                if not dst_img.exists():
                    import shutil
                    shutil.copy2(img_path, dst_img)

                # COCO image entry
                coco_images.append({
                    "id": processed,
                    "file_name": img_path.name,
                    "width": w,
                    "height": h,
                })

                # Guardar scores de confianza
                confidence_scores[img_path.name] = {
                    "mean_confidence": float(np.mean(img_confidences)) if img_confidences else 0.0,
                    "min_confidence": float(np.min(img_confidences)) if img_confidences else 0.0,
                    "num_detections": len(img_confidences),
                    "detections": img_confidences,
                }

                # Visualización
                if visualize and yolo_lines:
                    viz_image = image.copy()
                    if detections and detections.mask is not None:
                        for i in range(len(detections)):
                            if detections.class_id[i] < len(class_names):
                                cname = class_names[detections.class_id[i]]
                                color = CLASS_COLORS.get(cname, (128, 128, 128))
                                if detections.mask[i] is not None:
                                    mask_overlay = detections.mask[i].astype(np.uint8) * 255
                                    colored_mask = np.zeros_like(viz_image)
                                    colored_mask[:] = color
                                    alpha = 0.4
                                    mask_bool = mask_overlay > 0
                                    viz_image[mask_bool] = cv2.addWeighted(
                                        viz_image, 1 - alpha, colored_mask, alpha, 0
                                    )[mask_bool]

                    viz_path = viz_dir / f"viz_{img_path.name}"
                    cv2.imwrite(str(viz_path), viz_image)

                processed += 1

            except Exception as e:
                log.error("Error procesando %s: %s", img_path.name, e)
                errors += 1

            progress.advance(task)

    # Guardar scores de confianza
    scores_file = output_dir / "confidence_scores.json"
    with open(scores_file, "w") as f:
        json.dump(confidence_scores, f, indent=2)

    # Guardar COCO JSON
    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": cid, "name": cname, "supercategory": "damage"}
            for cname, cid in CLASS_IDS.items()
        ],
        "info": {
            "description": "Auto-labeled vehicle damage dataset",
            "date_created": datetime.now().isoformat(),
        },
    }
    coco_file = output_dir / "annotations_auto.json"
    with open(coco_file, "w") as f:
        json.dump(coco_output, f, indent=2)

    # Imprimir resumen
    print_summary(processed, errors, total_detections, confidence_scores)

    return confidence_scores


def print_summary(
    processed: int,
    errors: int,
    detections: dict,
    confidence_scores: dict,
):
    """Imprime resumen del auto-etiquetado."""
    table = Table(title="📊 Resumen de Auto-Etiquetado")
    table.add_column("Clase", style="cyan")
    table.add_column("Detecciones", justify="right", style="green")

    total = 0
    for class_name, count in sorted(detections.items()):
        table.add_row(class_name, f"{count:,}")
        total += count

    table.add_section()
    table.add_row("TOTAL", f"{total:,}", style="bold")

    console.print()
    console.print(table)
    console.print(f"\n  ✅ Procesadas: {processed}")
    console.print(f"  ❌ Errores: {errors}")

    if confidence_scores:
        all_means = [s["mean_confidence"] for s in confidence_scores.values() if s["mean_confidence"] > 0]
        if all_means:
            console.print(f"  📊 Confianza media global: {np.mean(all_means):.3f}")
            console.print(f"  📊 Confianza mínima: {np.min(all_means):.3f}")

    console.print(f"\n  ⚠️  Precisión esperada: ~70-80%. Revisar en CVAT.")


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-etiquetado con GroundingDINO + SAM (Autodistill)",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Directorio con imágenes sin anotar",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "auto_labeled",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Config de auto-labeling (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limitar a N imágenes (0 = todas)",
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Generar visualizaciones con masks (default: True)",
    )
    parser.add_argument(
        "--no-visualize", action="store_false", dest="visualize",
        help="No generar visualizaciones",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo: auto, cuda, cpu (default: auto)",
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
    console.print("[bold blue]  Auto-Etiquetado — GroundingDINO + SAM    [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Validar input
    if not args.input.exists():
        log.error("Directorio de entrada no existe: %s", args.input)
        sys.exit(1)

    # Cargar config
    config = yaml.safe_load(open(args.config))

    # Configurar dispositivo
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    console.print(f"  Dispositivo: [cyan]{device}[/]")
    console.print(f"  Entrada: [cyan]{args.input}[/]")
    console.print(f"  Salida: [cyan]{args.output}[/]")
    console.print()

    # Inicializar modelo
    model = init_model(config, device)

    # Ejecutar auto-labeling
    console.rule("[bold]Procesando imágenes[/]")
    run_auto_labeling(
        model=model,
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        limit=args.limit,
        visualize=args.visualize,
    )

    console.print(f"\n[bold green]✅ Auto-etiquetado completado[/]")
    console.print(f"   Labels YOLO: {args.output / 'labels'}")
    console.print(f"   COCO JSON: {args.output / 'annotations_auto.json'}")
    console.print(f"   Confianzas: {args.output / 'confidence_scores.json'}")
    console.print(f"\n   Siguiente paso: python scripts/prepare_for_review.py\n")


if __name__ == "__main__":
    main()
