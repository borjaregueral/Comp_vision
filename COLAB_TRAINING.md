# Entrenamiento en GPU (Google Colab)

El entrenamiento del modelo de daños (YOLOv11m-seg, 2 fases @ 1024px, hasta 300 epochs)
necesita GPU. En MPS local es inviable (días). Usa Colab (T4 gratis, o A100/L4 con Pro).

El notebook [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) lo automatiza de principio a fin.

---

## TL;DR

1. Sube el código (y opcionalmente los datos) a tu Google Drive.
2. Abre `notebooks/train_colab.ipynb` en Colab y pon el runtime en **GPU**.
3. Ejecuta las celdas en orden.
4. El modelo entrenado queda en tu Drive: `fotoperitacion/damage_seg/phase2_finetune/weights/best.pt`.

---

## 1. Preparar el código (local)

El código pesa <1 MB. Empaquétalo y súbelo a Drive en `MyDrive/fotoperitacion/`:

```bash
cd /Users/borja/Documents/Somniumrema/projects/Comp_vision
zip -r code.zip scripts configs requirements.txt
# → sube code.zip a Google Drive → MyDrive/fotoperitacion/code.zip
```

*(Alternativa: súbelo a GitHub y usa la celda de `git clone` del notebook — dímelo y te ayudo a crear el repo remoto con `gh`.)*

## 2. Conseguir los datos en Colab — elige UNA opción

**Opción A — Re-ingesta en Colab (recomendada, sin subir 3 GB).**
El notebook descarga VehiDE+CarDD y regenera `data/final` directamente en Colab.
Solo necesitas tu **`kaggle.json`** (Kaggle → *Settings* → *API* → *Create New Token*); el notebook te lo pedirá. CarDD es público (HuggingFace, sin credenciales).
Ventaja: regenera `configs/dataset.yaml` con las rutas correctas de Colab automáticamente.

**Opción B — Subir el `data/final` ya preparado (3.3 GB).**
```bash
cd /Users/borja/Documents/Somniumrema/projects/Comp_vision
zip -r data_final.zip data/final
# → sube data_final.zip a MyDrive/fotoperitacion/  (la subida de 3.3 GB tarda)
```
Descomenta la celda "Opción B" del notebook (descomprime + ajusta la ruta del `dataset.yaml`).

## 3. Entrenar

En Colab: `Archivo → Subir cuaderno → train_colab.ipynb` (o ábrelo desde Drive/GitHub).
`Entorno de ejecución → Cambiar tipo → GPU`. Ejecuta las celdas:

- **Sanity (opcional):** 3 epochs cortos para confirmar que corre en GPU.
- **Completo:** `python scripts/train.py --batch 16 --imgsz 1024` (Fase 1 congelada → Fase 2 fine-tuning).
  - T4 16 GB: si hay OOM, baja a `--batch 8`.
  - T4 gratis tiene límite de sesión (~12 h); para acortar usa `--epochs-phase2 100` o A100/L4 (Pro).

## 4. Recuperar el modelo

El notebook copia los resultados a tu Drive. Para usarlo en local:

```bash
# descarga desde Drive: fotoperitacion/damage_seg/phase2_finetune/weights/best.pt
mkdir -p runs/damage_seg/phase2_finetune/weights
# coloca ahí el best.pt descargado, luego:
python scripts/evaluate.py --model runs/damage_seg/phase2_finetune/weights/best.pt
python scripts/predict.py  --source path/a/imagen.jpg --model runs/damage_seg/phase2_finetune/weights/best.pt
python scripts/generate_report.py --source path/a/imagen.jpg
```

## 5. (Opcional) Modelo de partes — localización por zona

El notebook incluye una celda final que entrena el modelo `carparts-seg` (`train_parts.py`)
para asignar cada daño a una zona (front/rear/FL/FR/RL/RR). El `best.pt` resultante se usa con
`scripts/localize.py` o `generate_report.py --parts-model ...`.

---

**Notas**
- `requirements.txt` fija floors antiguos; en Colab basta `pip install ultralytics` (trae torch+CUDA correctos para la GPU). Evita forzar versiones viejas.
- Los checkpoints de Ultralytics se guardan cada época en `runs/`, así que un corte de sesión no pierde todo.
