#!/usr/bin/env python3
"""
train.py — Entrenamiento en 2 fases de YOLOv11-seg para detección de daños.

Fase 1: Backbone congelado (warm-up) — 20 epochs
Fase 2: Fine-tuning completo — 280 epochs con early stopping

Uso:
  python scripts/train.py
  python scripts/train.py --model yolo11s-seg.pt --imgsz 640 --batch 4
  python scripts/train.py --phase1-only --epochs-phase1 5
  python scripts/train.py --phase2-only --resume
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("train")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "damage_seg"


def detect_device() -> str:
    """Detecta el mejor dispositivo disponible."""
    try:
        import torch
    except ImportError:
        console.print("  [red]PyTorch no instalado[/]")
        return "cpu"

    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"  GPU detectada: [green]{gpu_name}[/] ({vram:.1f} GB VRAM)")
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            console.print("  Dispositivo: [green]Apple MPS[/]")
            return "mps"
        else:
            console.print("  Dispositivo: [yellow]CPU[/] (entrenamiento será lento)")
            return "cpu"
    except Exception as e:
        console.print(f"  [yellow]No se pudo consultar la GPU ({e}); usando CPU[/]")
        return "cpu"


def train_phase1(
    model_path: str,
    data_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    amp: bool = True,
    workers: int = 16,
    cache: "bool | str" = True,
    mask_ratio: int = 4,
    cls_pw: float = 0.0,
) -> Path:
    """Fase 1: Entrena con backbone congelado (warm-up)."""
    from ultralytics import YOLO

    console.print("\n[bold cyan]╔══════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║  FASE 1: Backbone Congelado (Warm-up)    ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/]\n")

    model = YOLO(model_path)

    console.print(f"  Modelo base:  [cyan]{model_path}[/]")
    console.print(f"  Epochs:       [cyan]{epochs}[/]")
    console.print(f"  Imagen:       [cyan]{imgsz}px[/]")
    console.print(f"  Batch:        [cyan]{batch}[/]")
    console.print(f"  Freeze:       [cyan]10 capas (backbone)[/]")
    console.print(f"  cls_pw:       [cyan]{cls_pw}[/]")
    console.print()

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="AdamW",
        lr0=0.01,
        freeze=10,           # Congelar backbone
        patience=0,          # Sin early stopping en fase 1
        amp=amp,
        device=device,
        workers=workers,
        cache=cache,
        mask_ratio=mask_ratio,
        cls_pw=cls_pw,
        project=project,
        name="phase1_frozen",
        exist_ok=True,
        verbose=True,
    )

    # Ruta REAL donde Ultralytics guardó (no la reconstruida): con project
    # relativo Ultralytics puede anidar bajo runs/segment/... y la reconstrucción
    # falla → el handoff fase1→fase2 se rompe en silencio. trainer.save_dir es la
    # fuente de verdad.
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / "phase1_frozen"))
    last_pt = save_dir / "weights" / "last.pt"

    if last_pt.exists():
        console.print(f"\n[green]✅ Fase 1 completada: {last_pt}[/]")
    else:
        console.print(f"[yellow]⚠ Weights en: {save_dir / 'weights'}[/]")

    return last_pt


def train_phase2(
    model_path: str,
    data_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    resume: bool = False,
    amp: bool = True,
    workers: int = 16,
    cache: "bool | str" = True,
    mask_ratio: int = 4,
    degrees: float = 15.0,
    scale: float = 0.5,
    flipud: float = 0.0,
    hsv_v: float = 0.4,
    close_mosaic: int = 10,
    cls_pw: float = 0.0,
) -> Path:
    """Fase 2: Fine-tuning completo con augmentaciones."""
    from ultralytics import YOLO

    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/]")
    console.print("[bold magenta]║  FASE 2: Fine-Tuning Completo           ║[/]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/]\n")

    model = YOLO(model_path)

    console.print(f"  Modelo:       [cyan]{model_path}[/]")
    console.print(f"  Epochs:       [cyan]{epochs}[/]")
    console.print(f"  Imagen:       [cyan]{imgsz}px[/]")
    console.print(f"  Batch:        [cyan]{batch}[/]")
    console.print(f"  Optimizer:    [cyan]AdamW[/]")
    console.print(f"  LR:           [cyan]0.001 → 0.00001[/]")
    console.print(f"  Patience:     [cyan]50 epochs[/]")
    console.print(f"  Mask ratio:   [cyan]{mask_ratio}[/]")
    console.print(f"  cls_pw:       [cyan]{cls_pw}[/]")
    console.print(f"  Aug (fina):   [cyan]degrees={degrees} scale={scale} flipud={flipud} "
                  f"hsv_v={hsv_v} close_mosaic={close_mosaic}[/]")
    console.print()

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        dropout=0.1,
        # ── Augmentaciones optimizadas para daños ──
        # degrees/scale/flipud/hsv_v/close_mosaic son configurables (Tier 1.3):
        # el down-scale de mosaic encoge cracks de 1-2px y cerrar mosaic al final
        # cierra el reality gap. Defaults = perfil baseline (A/B reproducible).
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,       # Copia daños a otras ubicaciones
        degrees=degrees,
        translate=0.2,
        scale=scale,
        close_mosaic=close_mosaic,
        flipud=flipud,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=hsv_v,
        # ── Control ──
        patience=50,
        amp=amp,
        device=device,
        workers=workers,
        cache=cache,
        mask_ratio=mask_ratio,
        cls_pw=cls_pw,
        project=project,
        name="phase2_finetune",
        exist_ok=True,
        resume=resume,
        verbose=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / "phase2_finetune"))
    best_pt = save_dir / "weights" / "best.pt"

    if best_pt.exists():
        console.print(f"\n[green]✅ Fase 2 completada: {best_pt}[/]")
    else:
        console.print(f"[yellow]⚠ Weights en: {save_dir / 'weights'}[/]")

    return best_pt


def print_training_summary(entries: "list[tuple[str, Path | None]]"):
    """Resumen a partir de las rutas REALES de cada fase (trainer.save_dir)."""
    table = Table(title="📋 Resumen de Entrenamiento")
    table.add_column("Fase", style="cyan")
    table.add_column("Weights")
    table.add_column("Estado", style="green")

    for phase_name, wpath in entries:
        ok = wpath is not None and wpath.exists()
        table.add_row(
            phase_name,
            str(wpath) if wpath else "—",
            "✅ OK" if ok else "❌ No encontrado",
        )

    console.print()
    console.print(table)


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento en 2 fases de YOLOv11-seg para daños en vehículos",
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help=f"Dataset YAML (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11m-seg.pt",
        help="Modelo base pretrained (default: yolo11m-seg.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Tamaño de imagen (default: 1024)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Procesos de carga de datos (default: 16). CPUs con muchos cores alimentan "
             "mejor la GPU y suben la utilización.",
    )
    parser.add_argument(
        "--cache", type=str, default="ram", choices=["ram", "disk", "none"],
        help="Cachear imágenes para no esperar al disco (default: ram). Ultralytics se "
             "desactiva solo si no cabe en RAM, así que es seguro por defecto.",
    )
    parser.add_argument(
        "--mask-ratio", type=int, default=4,
        help="Downsample de las máscaras GT para la loss de segmentación (default: 4, "
             "el de Ultralytics). Usa 1 para daños finos (rayones/grietas): con 4 una "
             "raya de 1-2px se vuelve sub-pixel y desaparece de la supervisión.",
    )
    # ── Augmentación fina-objeto (Tier 1.3) ─────────────────────────────
    # Defaults = perfil baseline (run 1.1). El segundo run los baja/sube para
    # atacar el suelo de scratch/crack: --scale 0.25 --degrees 10 --flipud 0.1
    #   --hsv-v 0.55 --close-mosaic 15. Solo afectan a la fase 2 (fine-tune).
    parser.add_argument(
        "--degrees", type=float, default=15.0,
        help="Rotación máxima en grados, fase 2 (default: 15). Tier 1.3 usa 10.",
    )
    parser.add_argument(
        "--scale", type=float, default=0.5,
        help="Ganancia de escala del mosaic, fase 2 (default: 0.5). El down-scale "
             "encoge cracks de 1-2px; Tier 1.3 lo baja a 0.25. [copy-paste 2012.07177]",
    )
    parser.add_argument(
        "--flipud", type=float, default=0.0,
        help="Prob. de volteo vertical, fase 2 (default: 0.0). Tier 1.3 usa 0.1: el "
             "daño no tiene 'arriba' canónico, da variedad sin romper realismo.",
    )
    parser.add_argument(
        "--hsv-v", type=float, default=0.4,
        help="Ganancia de brillo HSV-V, fase 2 (default: 0.4). Tier 1.3 sube a 0.55 "
             "para robustez ante luz/sombra de parking.",
    )
    parser.add_argument(
        "--close-mosaic", type=int, default=10,
        help="Epochs finales SIN mosaic, fase 2 (default: 10, el de Ultralytics). "
             "Tier 1.3 usa 15: cerrar el mosaic al final cierra el reality gap. "
             "[YOLOX close_mosaic 2107.08430]",
    )
    # ── Desbalance de clases (Tier 1.4) ─────────────────────────────────
    parser.add_argument(
        "--cls-pw", type=float, default=0.0,
        help="Peso por frecuencia inversa de clase en la loss de clasificación "
             "(default: 0.0 = desactivado, el de Ultralytics; rango válido [0,1]). "
             "Ultralytics calcula (1/frecuencia)**cls_pw normalizado a media 1.0 y lo "
             "aplica a la BCE de clase. Sweep del piloto: {0, 0.25, 0.5, 1.0} para subir "
             "las clases raras (broken_light/crack) sin tocar el muestreo. RFS suave "
             "[1908.03195] es el paso siguiente si esto no basta.",
    )
    parser.add_argument("--epochs-phase1", type=int, default=20, help="Epochs fase 1 (default: 20)")
    parser.add_argument("--epochs-phase2", type=int, default=280, help="Epochs fase 2 (default: 280)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Atajo: fija epochs para ambas fases (override de --epochs-phase1/--epochs-phase2). "
             "Útil para sanity checks, p.ej. --epochs 5 --phase1-only",
    )
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo (default: auto)")
    parser.add_argument("--project", type=str, default=str(DEFAULT_PROJECT))
    parser.add_argument("--phase1-only", action="store_true", help="Solo ejecutar fase 1")
    parser.add_argument("--phase2-only", action="store_true", help="Solo ejecutar fase 2")
    parser.add_argument("--resume", action="store_true", help="Reanudar entrenamiento interrumpido")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Precisión mixta (AMP). Usa --no-amp si la EMA da NaN/Inf al descongelar (fase 2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Atajo --epochs: override de los epochs por fase
    if args.epochs is not None:
        args.epochs_phase1 = args.epochs
        args.epochs_phase2 = args.epochs

    # cache: "none" → False (sin caché); "ram" → True; "disk" → "disk"
    cache_val: "bool | str" = {"ram": True, "disk": "disk", "none": False}[args.cache]

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Entrenamiento — Fotoperitación           [/]")
    console.print("[bold blue]  YOLOv11-seg · 2 Fases                    [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Detectar dispositivo
    if args.device == "auto":
        device = detect_device()
    else:
        device = args.device

    # Verificar dataset
    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Dataset no encontrado: %s", data_path)
        log.error("Ejecuta primero: python scripts/unify_to_yolo.py")
        sys.exit(1)

    # ── Fase 1 ────────────────────────────────────────────────────
    phase1_weights = None
    if not args.phase2_only:
        phase1_weights = train_phase1(
            model_path=args.model,
            data_path=args.data,
            epochs=args.epochs_phase1,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            amp=args.amp,
            workers=args.workers,
            cache=cache_val,
            mask_ratio=args.mask_ratio,
            cls_pw=args.cls_pw,
        )

    if args.phase1_only:
        print_training_summary([("phase1_frozen", phase1_weights)])
        console.print("\n[yellow]--phase1-only: Fase 2 omitida.[/]\n")
        return

    # ── Fase 2 ────────────────────────────────────────────────────
    if args.phase2_only:
        # Buscar weights de fase 1
        phase1_best = Path(args.project) / "phase1_frozen" / "weights" / "last.pt"
        if phase1_best.exists():
            phase2_model = str(phase1_best)
        else:
            log.warning("No se encontraron weights de fase 1. Usando modelo base.")
            phase2_model = args.model
    else:
        phase2_model = str(phase1_weights) if phase1_weights and phase1_weights.exists() else args.model

    best_model = train_phase2(
        model_path=phase2_model,
        data_path=args.data,
        epochs=args.epochs_phase2,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        resume=args.resume,
        amp=args.amp,
        workers=args.workers,
        cache=cache_val,
        mask_ratio=args.mask_ratio,
        degrees=args.degrees,
        scale=args.scale,
        flipud=args.flipud,
        hsv_v=args.hsv_v,
        close_mosaic=args.close_mosaic,
        cls_pw=args.cls_pw,
    )

    # ── Resumen ───────────────────────────────────────────────────
    summary_entries: "list[tuple[str, Path | None]]" = []
    if not args.phase2_only:
        summary_entries.append(("phase1_frozen", phase1_weights))
    summary_entries.append(("phase2_finetune", best_model))
    print_training_summary(summary_entries)

    console.print(f"\n[bold green]✅ Entrenamiento completado[/]")
    console.print(f"   Mejor modelo: {best_model}")
    console.print(f"\n   Siguientes pasos:")
    console.print(f"   python scripts/evaluate.py --model {best_model}")
    console.print(f"   python scripts/predict.py --source IMAGE --model {best_model}\n")


if __name__ == "__main__":
    main()
