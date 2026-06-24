#!/usr/bin/env bash
# ============================================================================
# setup_gpu.sh — Prepara una caja GPU (Linux/NVIDIA) y lanza el entrenamiento.
#
# Pensado para una máquina alquilada (RunPod, Vast.ai, Lambda, etc.) a la que
# entras por SSH. NO usa notebook. El entrenamiento corre en una sesión tmux,
# así que sigue aunque cierres el SSH o el portátil.
#
# USO (en la caja GPU):
#   git clone https://github.com/<TU_USUARIO>/Comp_vision.git
#   cd Comp_vision
#   # sube tu kaggle.json a  ~/.kaggle/kaggle.json   (para descargar VehiDE)
#
#   bash setup_gpu.sh                                  # legacy: VehiDE+CarDD → data/final
#   bash setup_gpu.sh --vehide4                        # PILOTO: VehiDE-only 4 clases (limpio)
#   bash setup_gpu.sh --vehide4 --epochs-phase2 100    # baseline más corto
#   bash setup_gpu.sh --vehide4 --batch 16             # GPU grande: sube el batch
#
#   # Tier 1.3 — segundo run, ataca el suelo de scratch/crack (finos). Pesado:
#   # mask-ratio 1 + 1280px → necesita A100 (40/80GB); en 24GB baja a --batch 2:
#   bash setup_gpu.sh --vehide4 --imgsz 1280 --mask-ratio 1 --batch 8 \
#        --scale 0.25 --degrees 10 --flipud 0.1 --hsv-v 0.55 --close-mosaic 15
#
# --vehide4 reconstruye el dataset limpio (sin fuga, 4 clases nativas) en
# data/final_vehide4 y entrena con configs/dataset_vehide4.yaml. Cualquier otro
# flag se pasa tal cual a train.py (y sobreescribe los defaults del modo).
#
# Si data/final[_vehide4]/images/train ya existe (lo subiste tú), salta la ingesta.
# ============================================================================
set -euo pipefail

# --- 0. Parseo de args: extraer --vehide4; el resto va a train.py -----------
MODE="default"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --vehide4) MODE="vehide4" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

if [ "$MODE" = "vehide4" ]; then
  DATA_CFG="configs/data_config_vehide4.yaml"
  DATASET_YAML="configs/dataset_vehide4.yaml"
  UNIFIED_DIR="data/unified_vehide4"
  FINAL_DIR="data/final_vehide4"
  DATASETS="vehide"
  PROJECT="runs/damage_seg_vehide4"
  # Defaults del piloto: --no-amp (seg_loss NaN bajo AMP) + batch 4 @ 1024px en 24GB.
  TRAIN_DEFAULTS=(--data "$DATASET_YAML" --model yolo11m-seg.pt --no-amp --batch 4 --imgsz 1024 --project "$PROJECT")
  echo "==> Modo: VehiDE-4 (piloto, dataset limpio sin fuga, 4 clases)"
else
  DATA_CFG="configs/data_config.yaml"
  DATASET_YAML="configs/dataset.yaml"
  UNIFIED_DIR="data/unified"
  FINAL_DIR="data/final"
  DATASETS="vehide,cardd"
  PROJECT="runs/damage_seg"
  TRAIN_DEFAULTS=()
  echo "==> Modo: legacy (VehiDE + CarDD → data/final)"
fi

echo "==> GPU detectada:"
nvidia-smi || { echo "!! No se detecta GPU NVIDIA. ¿Es esta una caja con GPU?"; exit 1; }

# --- 1. Dependencias --------------------------------------------------------
echo "==> Instalando dependencias (ultralytics trae torch+CUDA)..."
python -m pip install -q --upgrade pip
python -m pip install -q ultralytics kaggle huggingface_hub
if ! command -v tmux >/dev/null 2>&1; then
  echo "==> Instalando tmux..."
  (apt-get update -qq && apt-get install -y -qq tmux) \
    || (sudo apt-get update -qq && sudo apt-get install -y -qq tmux) \
    || { echo "!! No pude instalar tmux; instálalo manualmente."; exit 1; }
fi

# --- 2. Datos ---------------------------------------------------------------
if [ -d "$FINAL_DIR/images/train" ]; then
  echo "==> $FINAL_DIR ya existe; salto la ingesta."
else
  echo "==> Generando el dataset ($DATASETS)..."
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "!! Falta ~/.kaggle/kaggle.json (necesario para descargar VehiDE)."
    echo "   Kaggle → Settings → API → 'Create New Token' y súbelo a la caja:"
    echo "     mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
    exit 1
  fi
  chmod 600 "$HOME/.kaggle/kaggle.json"
  python scripts/download_datasets.py --datasets "$DATASETS" --config "$DATA_CFG"
  python scripts/unify_to_yolo.py --config "$DATA_CFG" --input "$UNIFIED_DIR" --output "$FINAL_DIR"
  if [ "$MODE" = "vehide4" ] && [ -f "$FINAL_DIR/leakage_audit.json" ]; then
    echo "==> Auditoría de fuga ($FINAL_DIR/leakage_audit.json):"
    grep -E '"(n_images|n_groups|cross_split_groups)"' "$FINAL_DIR/leakage_audit.json" || true
  fi
fi

# --- 2b. Apuntar el dataset.yaml a la ruta de ESTA máquina ------------------
# (el yaml versionado trae una ruta absoluta de otra máquina; si los datos se
#  subieron en vez de re-generarse, hay que corregir 'path').
DATASET_YAML="$DATASET_YAML" FINAL_DIR="$FINAL_DIR" python - <<'PY'
import os, yaml, pathlib
p = pathlib.Path(os.environ["DATASET_YAML"])
d = yaml.safe_load(p.read_text())
d["path"] = str(pathlib.Path(os.environ["FINAL_DIR"]).resolve())
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("==>", p, "path =", d["path"])
PY

# --- 3. Entrenamiento en tmux ----------------------------------------------
echo "==> Lanzando entrenamiento en una sesión tmux llamada 'train'..."
tmux kill-session -t train 2>/dev/null || true
TRAIN_CMD="python scripts/train.py ${TRAIN_DEFAULTS[*]:-} ${EXTRA_ARGS[*]:-}"
echo "    $TRAIN_CMD"
tmux new-session -d -s train "$TRAIN_CMD 2>&1 | tee train.log"

cat <<EOF

✅ Entrenando en segundo plano. La caja sigue trabajando aunque cierres el SSH.

   Ver en vivo:    tmux attach -t train      (salir sin parar:  Ctrl-b  luego  d)
   Ver el log:     tail -f train.log
   ¿Sigue vivo?:   tmux has-session -t train && echo running || echo done
   Modelo final:   $PROJECT/phase2_finetune/weights/best.pt

   Al terminar, evalúa (tabla por clase con mAP50 y mAP50-95 + JSON):
     python scripts/evaluate.py --model $PROJECT/phase2_finetune/weights/best.pt --data $DATASET_YAML

   Y/o descárgate best.pt a tu Mac:
     scp usuario@IP_DE_LA_CAJA:~/Comp_vision/$PROJECT/phase2_finetune/weights/best.pt .
EOF
