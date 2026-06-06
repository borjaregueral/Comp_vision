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
#   bash setup_gpu.sh                                  # run completo (2 fases)
#   bash setup_gpu.sh --batch 16 --imgsz 1024          # flags extra → train.py
#   bash setup_gpu.sh --batch 16 --epochs-phase2 100   # versión más corta
#
# Si ya tienes data/final en la caja (la subiste tú), el script salta la ingesta.
# ============================================================================
set -euo pipefail

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
if [ -d data/final/images/train ]; then
  echo "==> data/final ya existe; salto la ingesta."
else
  echo "==> Generando el dataset (VehiDE + CarDD)..."
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "!! Falta ~/.kaggle/kaggle.json (necesario para descargar VehiDE)."
    echo "   Kaggle → Settings → API → 'Create New Token' y súbelo a la caja:"
    echo "     mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
    echo "   (CarDD se descarga sin credenciales desde HuggingFace.)"
    exit 1
  fi
  chmod 600 "$HOME/.kaggle/kaggle.json"
  python scripts/download_datasets.py --datasets vehide,cardd
  python scripts/unify_to_yolo.py     # genera data/final + configs/dataset.yaml
fi

# --- 2b. Apuntar dataset.yaml a la ruta de ESTA máquina --------------------
# (el dataset.yaml versionado trae una ruta absoluta de otra máquina; si los
#  datos se subieron en vez de re-generarse, hay que corregir 'path').
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/dataset.yaml")
d = yaml.safe_load(p.read_text())
d["path"] = str(pathlib.Path("data/final").resolve())
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("==> dataset.yaml path =", d["path"])
PY

# --- 3. Entrenamiento en tmux ----------------------------------------------
echo "==> Lanzando entrenamiento en una sesión tmux llamada 'train'..."
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "python scripts/train.py $* 2>&1 | tee train.log"

cat <<'EOF'

✅ Entrenando en segundo plano. La caja sigue trabajando aunque cierres el SSH.

   Ver en vivo:    tmux attach -t train      (salir sin parar:  Ctrl-b  luego  d)
   Ver el log:     tail -f train.log
   ¿Sigue vivo?:   tmux has-session -t train && echo running || echo done
   Modelo final:   runs/damage_seg/phase2_finetune/weights/best.pt

   Al terminar, descárgate best.pt a tu Mac, p.ej.:
     scp usuario@IP_DE_LA_CAJA:~/Comp_vision/runs/damage_seg/phase2_finetune/weights/best.pt .
EOF
