#!/usr/bin/env python3
"""
evaluate.py — Evaluación completa del modelo sobre el test set.

Calcula mAP, precision, recall, F1 por clase. Genera matriz de confusión,
curvas PR, y comparaciones visuales GT vs predicción.

Uso:
  python scripts/evaluate.py --model runs/damage_seg/phase2_finetune/weights/best.pt
  python scripts/evaluate.py --model best.pt --data configs/dataset.yaml --samples 30
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("evaluate")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "damage_seg" / "phase2_finetune" / "weights" / "best.pt"

# Defaults (v1 4-clases); en main() se SOBREESCRIBEN con model.names para ser
# agnósticos a la taxonomía (4 clases v1 o 6 clases v2 o la que sea).
CLASS_NAMES = {0: "dent", 1: "scratch", 2: "crack", 3: "broken_light"}

# Paleta BGR ciclable para cualquier nº de clases
_PALETTE_BGR = [
    (68, 68, 255), (0, 215, 255), (255, 136, 68), (0, 165, 255),
    (128, 0, 128), (255, 68, 170), (0, 200, 0), (200, 200, 0),
]


def _build_colors(class_names: dict) -> dict:
    return {cid: _PALETTE_BGR[i % len(_PALETTE_BGR)] for i, cid in enumerate(sorted(class_names))}


CLASS_COLORS_BGR = _build_colors(CLASS_NAMES)


def run_validation(model, data_yaml: str, device: str = "auto", tta: bool = False) -> dict:
    """Ejecuta validación de Ultralytics y extrae métricas.

    tta=True activa augment (test-time augmentation): más lento pero típicamente
    +0.5–1.5 mAP (Tier 1.6). La val barre la confianza en un solo pase, así que de
    aquí salen las curvas P/R/F1 que usa analyze_operating_point.
    """
    console.print("[bold]Ejecutando validación en test set...[/]\n")
    if tta:
        console.print("  [cyan]TTA activado (augment=True): más lento, +0.5–1.5 mAP esperado[/]\n")

    metrics = model.val(
        data=data_yaml,
        split="test",
        verbose=True,
        device=device if device != "auto" else None,
        augment=tta,
    )

    return metrics


def analyze_operating_point(metrics, beta: float = 2.0) -> dict:
    """Recomienda el umbral de confianza operativo para MÁSCARA (Tier 1.6).

    Del barrido de confianza de la val (un solo pase) toma las curvas de Ultralytics
    (px = eje de confianza; p_curve/r_curve/f1_curve = (nc, n)) y devuelve:
      • f1_peak       — conf del pico de F1 (punto P/R balanceado).
      • recall_biased — conf que maximiza F-beta (beta=2 → pondera recall el doble):
                        el punto de despliegue del piloto ("no perder daño").
    Promedia sobre las clases presentes para dar un umbral global desplegable.
    """
    seg = getattr(metrics, "seg", None)
    out: dict = {"available": False}
    if seg is None:
        return out

    px = np.asarray(getattr(seg, "px", []), dtype=float)         # (n,) eje de confianza
    p_c = np.asarray(getattr(seg, "p_curve", []), dtype=float)   # (nc, n)
    r_c = np.asarray(getattr(seg, "r_curve", []), dtype=float)
    f1_c = np.asarray(getattr(seg, "f1_curve", []), dtype=float)
    if px.size == 0 or f1_c.size == 0 or f1_c.ndim != 2 or p_c.shape != r_c.shape:
        return out

    # Media sobre clases presentes → curvas globales de longitud n.
    p, r, f1 = p_c.mean(0), r_c.mean(0), f1_c.mean(0)
    b2 = beta * beta
    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)

    i_f1, i_fb = int(np.argmax(f1)), int(np.argmax(fbeta))
    out.update({
        "available": True,
        "beta": beta,
        "f1_peak": {
            "conf": float(px[i_f1]), "precision": float(p[i_f1]),
            "recall": float(r[i_f1]), "f1": float(f1[i_f1]),
        },
        "recall_biased": {
            "conf": float(px[i_fb]), "precision": float(p[i_fb]),
            "recall": float(r[i_fb]), "f1": float(f1[i_fb]), "fbeta": float(fbeta[i_fb]),
        },
    })
    return out


def print_operating_point(op: dict):
    """Imprime la recomendación de umbral operativo de máscara."""
    if not op.get("available"):
        console.print("\n[yellow]Umbral operativo: curvas de máscara no disponibles "
                      "(¿sin detecciones en test?).[/]")
        return
    table = Table(title="🎯 Umbral operativo (máscara)")
    table.add_column("Punto", style="cyan")
    table.add_column("conf", justify="right", style="green")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right", style="green")
    table.add_column("F1", justify="right")
    f1p, rb = op["f1_peak"], op["recall_biased"]
    table.add_row("Pico F1", f"{f1p['conf']:.3f}", f"{f1p['precision']:.3f}",
                  f"{f1p['recall']:.3f}", f"{f1p['f1']:.3f}")
    table.add_row(f"Sesgado a recall (F{op['beta']:.0f})", f"{rb['conf']:.3f}",
                  f"{rb['precision']:.3f}", f"{rb['recall']:.3f}", f"{rb['f1']:.3f}")
    console.print()
    console.print(table)
    console.print(f"  [dim]Despliegue: conf≈{rb['conf']:.3f} (sesgado a recall, no perder "
                  f"daño). Pico F1 en conf≈{f1p['conf']:.3f}.[/]")


def extract_metrics(metrics) -> dict:
    """Extrae y estructura las métricas de validación."""
    results = {
        "box": {
            "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
            "mAP50_95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
        },
        "mask": {
            "mAP50": float(metrics.seg.map50) if hasattr(metrics, 'seg') and hasattr(metrics.seg, 'map50') else 0,
            "mAP50_95": float(metrics.seg.map) if hasattr(metrics, 'seg') and hasattr(metrics.seg, 'map') else 0,
        },
        "per_class": {},
    }

    # Métricas por clase
    # IMPORTANTE: Ultralytics ordena p/r/ap50/ap por las clases REALMENTE
    # presentes en el eval set (no una entrada por id de clase). El índice real
    # de cada fila está en `ap_class_index`. Indexar por id absoluto desalinea
    # las métricas. Mapeamos posición → id de clase vía ap_class_index.
    box_idx = [int(c) for c in getattr(metrics.box, "ap_class_index", [])]
    seg = metrics.seg if hasattr(metrics, "seg") else None
    seg_idx = [int(c) for c in getattr(seg, "ap_class_index", [])] if seg is not None else []

    for cid, class_name in CLASS_NAMES.items():
        class_metrics = {}
        try:
            if cid in box_idx:
                # class_result devuelve (p, r, ap50, ap) — el 4º es el mAP50-95 de
                # esa clase. Lo guardamos: la puerta de decisión del piloto exige
                # mAP50 *y* mAP50-95 POR CLASE (la media global esconde scratch/crack).
                p, r, ap50, ap = metrics.box.class_result(box_idx.index(cid))
                class_metrics["box_ap50"] = float(ap50)
                class_metrics["box_ap50_95"] = float(ap)
                class_metrics["precision"] = float(p)
                class_metrics["recall"] = float(r)
                pr, rc = class_metrics["precision"], class_metrics["recall"]
                class_metrics["f1"] = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
            if seg is not None and cid in seg_idx:
                _, _, seg_ap50, seg_ap = seg.class_result(seg_idx.index(cid))
                class_metrics["mask_ap50"] = float(seg_ap50)
                class_metrics["mask_ap50_95"] = float(seg_ap)
        except (IndexError, AttributeError):
            pass

        results["per_class"][class_name] = class_metrics

    return results


def print_metrics_table(results: dict):
    """Imprime las métricas como tablas rich."""
    # Tabla global
    table = Table(title="📊 Métricas Globales")
    table.add_column("Métrica", style="cyan")
    table.add_column("Boxes", justify="right", style="green")
    table.add_column("Masks", justify="right", style="green")

    table.add_row(
        "mAP@50",
        f"{results['box']['mAP50']:.4f}",
        f"{results['mask']['mAP50']:.4f}",
    )
    table.add_row(
        "mAP@50:95",
        f"{results['box']['mAP50_95']:.4f}",
        f"{results['mask']['mAP50_95']:.4f}",
    )
    console.print(table)

    # Tabla por clase
    table2 = Table(title="📊 Métricas por Clase")
    table2.add_column("Clase", style="cyan")
    table2.add_column("AP@50 (box)", justify="right")
    table2.add_column("AP@50:95 (box)", justify="right")
    table2.add_column("AP@50 (mask)", justify="right")
    table2.add_column("AP@50:95 (mask)", justify="right")
    table2.add_column("Precision", justify="right")
    table2.add_column("Recall", justify="right")
    table2.add_column("F1", justify="right", style="green")

    for class_name, m in results["per_class"].items():
        f1 = m.get("f1", 0)
        f1_color = "green" if f1 > 0.7 else ("yellow" if f1 > 0.4 else "red")
        table2.add_row(
            class_name,
            f"{m.get('box_ap50', 0):.4f}",
            f"{m.get('box_ap50_95', 0):.4f}",
            f"{m.get('mask_ap50', 0):.4f}",
            f"{m.get('mask_ap50_95', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"[{f1_color}]{f1:.4f}[/]",
        )

    console.print()
    console.print(table2)


def generate_comparison_images(
    model,
    data_yaml: str,
    output_dir: Path,
    n_samples: int = 20,
):
    """Genera imágenes comparativas GT vs Predicción."""
    import yaml

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    test_images_dir = Path(data_cfg["path"]) / data_cfg["test"]
    test_labels_dir = Path(data_cfg["path"]) / data_cfg["test"].replace("images", "labels")

    if not test_images_dir.exists():
        log.warning("Test images dir not found: %s", test_images_dir)
        return

    # Obtener imágenes de test
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_files:
        log.warning("No test images found")
        return

    # Seleccionar muestra aleatoria
    random.seed(42)
    samples = random.sample(image_files, min(n_samples, len(image_files)))

    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Generando {len(samples)} comparaciones GT vs Predicción...[/]")

    for img_path in samples:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # Crear panel: [GT | Predicción]
        gt_img = image.copy()
        pred_img = image.copy()

        # Dibujar ground truth
        label_file = test_labels_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    # Desnormalizar
                    points = []
                    for i in range(0, len(coords), 2):
                        px = int(coords[i] * w)
                        py = int(coords[i + 1] * h)
                        points.append([px, py])

                    if points:
                        pts = np.array(points, dtype=np.int32)
                        color = CLASS_COLORS_BGR.get(class_id, (128, 128, 128))
                        cv2.polylines(gt_img, [pts], True, color, 2)
                        # Fill semitransparente
                        overlay = gt_img.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        gt_img = cv2.addWeighted(overlay, 0.3, gt_img, 0.7, 0)

        # Dibujar predicciones
        results = model.predict(str(img_path), verbose=False, conf=0.25)
        if results and results[0].masks is not None:
            for i in range(len(results[0].masks)):
                class_id = int(results[0].boxes.cls[i])
                color = CLASS_COLORS_BGR.get(class_id, (128, 128, 128))

                mask = results[0].masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))

                overlay = pred_img.copy()
                overlay[mask_resized > 0] = color
                pred_img = cv2.addWeighted(overlay, 0.3, pred_img, 0.7, 0)

                # Bbox
                bbox = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                cv2.rectangle(pred_img, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                conf = float(results[0].boxes.conf[i])
                label = f"{CLASS_NAMES.get(class_id, '?')} {conf:.2f}"
                cv2.putText(pred_img, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Añadir títulos
        cv2.putText(gt_img, "GROUND TRUTH", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_img, "PREDICTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenar
        separator = np.ones((h, 3, 3), dtype=np.uint8) * 255
        comparison = np.hstack([gt_img, separator, pred_img])

        # Guardar
        out_path = comparisons_dir / f"compare_{img_path.name}"
        cv2.imwrite(str(out_path), comparison)

    log.info("Comparaciones guardadas en %s", comparisons_dir)


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación completa del modelo de detección de daños",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help="Modelo entrenado (.pt)",
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation_results")
    parser.add_argument("--samples", type=int, default=20, help="Comparaciones GT vs pred")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--tta", action="store_true",
        help="Test-time augmentation (augment=True) en la val: más lento, +0.5–1.5 mAP "
             "esperado (Tier 1.6).",
    )
    parser.add_argument(
        "--beta", type=float, default=2.0,
        help="Beta del F-beta para el umbral operativo sesgado a recall (default: 2.0 "
             "→ recall pesa el doble; no perder daño). Tier 1.6.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Evaluación del Modelo                    [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    from ultralytics import YOLO

    if not Path(args.model).exists():
        log.error("Modelo no encontrado: %s", args.model)
        sys.exit(1)

    model = YOLO(args.model)

    # Taxonomía agnóstica: las clases salen del propio modelo (v1 4-cls / v2 6-cls)
    global CLASS_NAMES, CLASS_COLORS_BGR
    if getattr(model, "names", None):
        CLASS_NAMES = {int(k): v for k, v in model.names.items()}
        CLASS_COLORS_BGR = _build_colors(CLASS_NAMES)
        log.info("Clases del modelo (%d): %s", len(CLASS_NAMES),
                 ", ".join(CLASS_NAMES[c] for c in sorted(CLASS_NAMES)))

    args.output.mkdir(parents=True, exist_ok=True)

    # Validación
    console.rule("[bold]Validación en Test Set[/]")
    metrics = run_validation(model, args.data, args.device, tta=args.tta)

    # Extraer y mostrar métricas
    results = extract_metrics(metrics)
    print_metrics_table(results)

    # Umbral operativo (Tier 1.6): pico F1 + punto sesgado a recall de la máscara
    operating_point = analyze_operating_point(metrics, beta=args.beta)
    print_operating_point(operating_point)
    results["operating_point"] = operating_point
    results["tta"] = bool(args.tta)

    # Guardar métricas JSON
    metrics_file = args.output / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Métricas guardadas: %s", metrics_file)

    # Generar comparaciones
    console.rule("[bold]Comparaciones GT vs Predicción[/]")
    generate_comparison_images(model, args.data, args.output, args.samples)

    # Copiar plots de Ultralytics si existen
    val_dir = Path(args.model).parent.parent
    for plot_name in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = val_dir / plot_name
        if src.exists():
            import shutil
            shutil.copy2(src, args.output / plot_name)

    console.print(f"\n[bold green]✅ Evaluación completada[/]")
    console.print(f"   Resultados en: {args.output}")
    console.print(f"   - metrics.json")
    console.print(f"   - comparisons/")
    console.print(f"   - confusion_matrix.png (si disponible)")
    console.print(f"   - PR_curve.png (si disponible)\n")


if __name__ == "__main__":
    main()
