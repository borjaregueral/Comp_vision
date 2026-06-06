#!/usr/bin/env python3
"""
predict.py — Inferencia de daños en vehículos con visualización y salida JSON.

Ejecuta el modelo entrenado sobre imágenes y genera visualizaciones con
máscaras coloreadas por tipo de daño + informe JSON estructurado.

Uso:
  python scripts/predict.py --source path/to/image.jpg --model runs/damage_seg/phase2_finetune/weights/best.pt
  python scripts/predict.py --source path/to/dir/ --save-json --save-viz
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("predict")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = {0: "dent", 1: "scratch", 2: "crack", 3: "broken_light"}
CLASS_NAMES_ES = {0: "Abolladura", 1: "Arañazo", 2: "Grieta", 3: "Faro/Piloto roto"}

# Colores BGR para OpenCV
CLASS_COLORS_BGR = {
    0: (68, 68, 255),      # Rojo (dent)
    1: (0, 215, 255),      # Amarillo (scratch)
    2: (255, 136, 68),     # Azul (crack)
    3: (255, 68, 170),     # Morado (broken_light)
}

CLASS_COLORS_HEX = {
    0: "#FF4444",
    1: "#FFD700",
    2: "#4488FF",
    3: "#AA44FF",
}


def find_images(source: Path) -> list[Path]:
    """Encuentra imágenes en la fuente (archivo o directorio)."""
    if source.is_file():
        return [source]

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for ext in extensions:
        images.extend(source.rglob(f"*{ext}"))
        images.extend(source.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_inference(
    model,
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
    calibrator=None,
) -> dict:
    """Ejecuta inferencia en una imagen y estructura los resultados.
    
    Returns:
        Dict con la estructura del informe de daños.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    if not results:
        return {"image": image_path.name, "damages": [], "summary": {}}

    result = results[0]
    img_h, img_w = result.orig_shape
    damages = []

    if result.masks is not None and len(result.masks) > 0:
        for i in range(len(result.masks)):
            # Clase y confianza
            class_id = int(result.boxes.cls[i])
            confidence = float(result.boxes.conf[i])

            # Bounding box
            bbox = result.boxes.xyxy[i].cpu().numpy().tolist()
            bbox = [round(v, 1) for v in bbox]

            # Máscara → polígono
            mask = result.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Área del daño
            area_px = int(np.sum(mask_resized > 0))
            area_pct = round(area_px / (img_w * img_h) * 100, 2)

            # Extraer polígono de la máscara
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Tomar el contorno más grande
            polygon = []
            if contours:
                largest = max(contours, key=cv2.contourArea)
                # Simplificar polígono
                epsilon = 0.002 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                polygon = approx.reshape(-1, 2).tolist()

            damages.append({
                "class": CLASS_NAMES.get(class_id, "unknown"),
                "class_es": CLASS_NAMES_ES.get(class_id, "Desconocido"),
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "area_px": area_px,
                "area_pct": area_pct,
                "bbox": bbox,
                "mask_polygon": polygon,
            })

    # Calibración de confianza (T3.4): mapea conf cruda → calibrada SIN reentrenar.
    # Conserva la cruda en confidence_raw para auditoría.
    if calibrator is not None and damages:
        raws = [d["confidence"] for d in damages]
        cals = calibrator.transform(raws)
        for d, raw, cal in zip(damages, raws, cals):
            d["confidence_raw"] = raw
            d["confidence"] = round(float(cal), 4)

    # Resumen
    total_area = sum(d["area_pct"] for d in damages)
    damage_types = list(set(d["class"] for d in damages))

    # Severidad PRELIMINAR (solo visual, por % de área en la imagen). Sustituye
    # los antiguos magic numbers; umbrales en business_rules/severity_matrix.yaml.
    # NO es la severidad económica autoritativa: esa la calcula scripts/severity.py
    # (compute_severity) en el pipeline, por pieza + zona + coste.
    from severity import preliminary_visual_severity
    severity = preliminary_visual_severity(total_area)
    severity_en = {"Leve": "Minor", "Moderado": "Moderate", "Severo": "Severe"}[severity]

    report = {
        "image": image_path.name,
        "image_path": str(image_path),
        "image_size": [img_w, img_h],
        "timestamp": datetime.now().isoformat(),
        "damages": damages,
        "summary": {
            "total_damages": len(damages),
            "total_damage_area_pct": round(total_area, 2),
            "damage_types": damage_types,
            "severity": severity,
            "severity_en": severity_en,
        },
    }

    return report


def draw_visualization(
    image_path: Path,
    report: dict,
    alpha: float = 0.4,
) -> np.ndarray:
    """Dibuja las máscaras y bboxes sobre la imagen."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    overlay = image.copy()

    for damage in report["damages"]:
        class_id = damage["class_id"]
        color = CLASS_COLORS_BGR.get(class_id, (128, 128, 128))
        confidence = damage["confidence"]
        class_name = damage["class"]

        # Dibujar polígono relleno
        polygon = damage.get("mask_polygon", [])
        if polygon:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

        # Dibujar bbox
        bbox = damage["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(
            image, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    # Blend masks
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Leyenda
    legend_y = 30
    for cid, cname in CLASS_NAMES.items():
        color = CLASS_COLORS_BGR[cid]
        cv2.rectangle(result, (10, legend_y - 15), (30, legend_y), color, -1)
        cv2.putText(
            result, cname, (35, legend_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        legend_y += 25

    return result


def print_report_table(report: dict):
    """Imprime el informe de daños como tabla rich."""
    table = Table(title=f"🔍 Daños detectados: {report['image']}")
    table.add_column("#", style="dim")
    table.add_column("Tipo", style="cyan")
    table.add_column("Confianza", justify="right")
    table.add_column("Área (px)", justify="right")
    table.add_column("Área (%)", justify="right")

    for i, d in enumerate(report["damages"], 1):
        conf_style = "green" if d["confidence"] > 0.7 else ("yellow" if d["confidence"] > 0.4 else "red")
        table.add_row(
            str(i),
            d["class_es"],
            f"[{conf_style}]{d['confidence']:.3f}[/]",
            f"{d['area_px']:,}",
            f"{d['area_pct']:.2f}%",
        )

    console.print(table)

    summary = report["summary"]
    severity = summary["severity"]
    sev_color = "green" if severity == "Leve" else ("yellow" if severity == "Moderado" else "red")

    console.print(f"  Total daños: [bold]{summary['total_damages']}[/]")
    console.print(f"  Área total dañada: [bold]{summary['total_damage_area_pct']:.2f}%[/]")
    console.print(f"  Severidad: [{sev_color}][bold]{severity}[/][/]")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de daños en vehículos con visualización",
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Imagen o directorio de imágenes",
    )
    parser.add_argument(
        "--model", type=str,
        default=str(PROJECT_ROOT / "runs" / "damage_seg" / "phase2_finetune" / "weights" / "best.pt"),
        help="Ruta al modelo entrenado (.pt)",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="Umbral IoU NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Tamaño de imagen (default: 1024)")
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--save-viz", action="store_true", default=True)
    parser.add_argument("--no-save-json", action="store_false", dest="save_json")
    parser.add_argument("--no-save-viz", action="store_false", dest="save_viz")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibrate", action="store_true",
                        help="Aplicar calibrador de confianza (T3.4) sin reentrenar el modelo.")
    parser.add_argument("--calibrator", type=str, default=None,
                        help="Ruta al calibrador .pkl (default: calibration.yaml > calibrator_path).")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Inferencia — Detección de Daños          [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Cargar modelo
    from ultralytics import YOLO

    if not Path(args.model).exists():
        log.error("Modelo no encontrado: %s", args.model)
        log.error("Entrena primero: python scripts/train.py")
        sys.exit(1)

    model = YOLO(args.model)
    console.print(f"  Modelo: [cyan]{args.model}[/]")

    # Calibrador de confianza (opcional, T3.4)
    calibrator = None
    if args.calibrate:
        from calibrate_confidence import load_calibrator, load_config as _cal_cfg
        cal_path = args.calibrator or (PROJECT_ROOT / _cal_cfg().get("calibrator_path"))
        calibrator = load_calibrator(cal_path)
        console.print(f"  Calibrador: [cyan]{cal_path}[/]")

    # Encontrar imágenes
    images = find_images(args.source)
    if not images:
        log.error("No se encontraron imágenes en: %s", args.source)
        sys.exit(1)

    console.print(f"  Imágenes: [cyan]{len(images)}[/]")
    console.print()

    # Crear directorio de salida
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        (args.output / "visualizations").mkdir(exist_ok=True)

    # Procesar
    all_reports = []
    for img_path in images:
        console.rule(f"[dim]{img_path.name}[/]")

        report = run_inference(model, img_path, args.conf, args.iou, args.imgsz, calibrator=calibrator)
        all_reports.append(report)

        if report["damages"]:
            print_report_table(report)
        else:
            console.print("  [dim]Sin daños detectados[/]")

        # Guardar visualización
        if args.save_viz and report["damages"]:
            viz = draw_visualization(img_path, report)
            if viz is not None:
                viz_path = args.output / "visualizations" / f"pred_{img_path.name}"
                cv2.imwrite(str(viz_path), viz)

    # Guardar JSON
    if args.save_json:
        json_path = args.output / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        log.info("Predicciones guardadas: %s", json_path)

    # Resumen global
    console.print()
    console.rule("[bold]Resumen Global[/]")
    total_damages = sum(r["summary"]["total_damages"] for r in all_reports)
    images_with_damage = sum(1 for r in all_reports if r["damages"])
    console.print(f"  Imágenes procesadas:    [bold]{len(all_reports)}[/]")
    console.print(f"  Imágenes con daños:     [bold]{images_with_damage}[/]")
    console.print(f"  Total daños detectados: [bold]{total_damages}[/]")
    console.print(f"\n  Resultados en: [cyan]{args.output}[/]\n")


if __name__ == "__main__":
    main()
