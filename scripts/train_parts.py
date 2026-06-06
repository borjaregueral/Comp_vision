#!/usr/bin/env python3
"""
train_parts.py — Entrena el modelo de SEGMENTACIÓN DE PARTES del vehículo.

Usa el dataset Ultralytics `carparts-seg` (23 clases con partes diferenciadas
por lado: front_left_door, back_right_light, ...). Ultralytics lo descarga
automáticamente la primera vez. El modelo resultante alimenta a localize.py
para asignar cada daño a una zona (front/rear/front_left/...).

Uso:
  python scripts/train_parts.py                       # entrenamiento completo
  python scripts/train_parts.py --epochs 5 --imgsz 640 --batch 8   # sanity check
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger("train_parts")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "parts_seg"

# Reutiliza la detección de dispositivo de train.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import detect_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de partes (carparts-seg) para localización por zona",
    )
    parser.add_argument("--data", type=str, default="carparts-seg.yaml",
                        help="Dataset Ultralytics (default: carparts-seg.yaml, se descarga solo)")
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt",
                        help="Modelo base pretrained (default: yolo11m-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping (default: 20)")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo (default: auto)")
    parser.add_argument("--project", type=str, default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", type=str, default="train")
    return parser.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()
    device = detect_device() if args.device == "auto" else args.device

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Entrenamiento — Partes del Vehículo      [/]")
    console.print("[bold blue]  YOLOv11-seg · carparts-seg (23 clases)   [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")
    console.print(f"  Dataset:  [cyan]{args.data}[/] (se descarga si no está presente)")
    console.print(f"  Modelo:   [cyan]{args.model}[/]")
    console.print(f"  Epochs:   [cyan]{args.epochs}[/]  Imagen: {args.imgsz}px  Batch: {args.batch}")
    console.print(f"  Salida:   [cyan]{args.project}/{args.name}[/]\n")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentaciones moderadas (las partes son objetos grandes y estables)
        mosaic=1.0,
        fliplr=0.0,   # OJO: NO voltear horizontal — invertiría izquierda/derecha
        flipud=0.0,
        degrees=5.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    console.print(f"\n[bold green]✅ Modelo de partes entrenado[/]")
    console.print(f"   Pesos: {best}")
    console.print(f"   Úsalo con: python scripts/localize.py --source IMG --parts-model {best}\n")


if __name__ == "__main__":
    main()
