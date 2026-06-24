#!/usr/bin/env python3
"""
make_rfs_list.py — Repeat-Factor Sampling (RFS) para el desbalance (Tier 1.4).

PROBLEMA: en VehiDE-4 las clases raras (broken_light, crack) aparecen en muchas
menos imágenes que scratch/dent. cls_pw pondera la loss; RFS ataca el otro frente
—el MUESTREO— sobre-representando las imágenes que contienen clases raras.

CÓMO (LVIS, [1908.03195], "suave" con t≈0.001, sin tocar internals de Ultralytics):
  • f(c) = fracción de imágenes de train que contienen la clase c (frecuencia de
    IMAGEN, no de instancia).
  • r(c) = max(1, sqrt(t / f(c)))     — factor de repetición por clase.
  • r(i) = max_{c en i} r(c)          — por imagen: la clase más rara que contiene.
  • la imagen i se repite round_estocástico(r(i)) veces en la lista de train.

Ultralytics lee la lista de train con sorted() y SIN deduplicar (data/base.py), así
que basta escribir un train_rfs.txt con cada ruta repetida r(i) veces y apuntar el
`train:` del dataset.yaml a ese fichero. No hace falta sampler propio ni regenerar
imágenes.

Uso (en la caja, sobre el dataset ya construido):
  python scripts/make_rfs_list.py --final-dir data/final_vehide4 --t 0.001
  # → escribe data/final_vehide4/train_rfs.txt + rfs_audit.json e imprime el cambio
  #   a hacer en el dataset.yaml:  train: train_rfs.txt
  python scripts/train.py --data configs/dataset_vehide4_rfs.yaml ...
"""

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rfs")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ── Núcleo RFS (puro, testeable sin ficheros) ─────────────────────────────

def class_image_frequencies(image_classes: "dict[str, set[int]]") -> "dict[int, float]":
    """f(c) = (# imágenes que contienen c) / (# imágenes totales)."""
    n = len(image_classes) or 1
    counts: Counter = Counter()
    for classes in image_classes.values():
        for c in set(classes):
            counts[c] += 1
    return {c: counts[c] / n for c in counts}


def repeat_factor_per_class(freqs: "dict[int, float]", t: float) -> "dict[int, float]":
    """r(c) = max(1, sqrt(t / f(c))). t es el umbral 'suave' (≈0.001)."""
    return {c: max(1.0, math.sqrt(t / f)) for c, f in freqs.items() if f > 0}


def repeat_factor_per_image(
    image_classes: "dict[str, set[int]]", rc: "dict[int, float]"
) -> "dict[str, float]":
    """r(i) = max factor de las clases en i; 1.0 si la imagen no tiene etiquetas."""
    out: "dict[str, float]" = {}
    for img, classes in image_classes.items():
        out[img] = max((rc.get(c, 1.0) for c in set(classes)), default=1.0)
    return out


def stochastic_count(r: float, u: float) -> int:
    """floor(r) + 1 con probabilidad frac(r). u es un uniforme[0,1).

    Redondeo estocástico: en expectativa repite exactamente r(i) veces, sin sesgar
    los factores fraccionarios siempre hacia arriba o hacia abajo.
    """
    fl = math.floor(r)
    return int(fl) + (1 if u < (r - fl) else 0)


# ── I/O ───────────────────────────────────────────────────────────────────

def read_image_classes(labels_dir: Path, images_dir: Path) -> "dict[str, set[int]]":
    """Lee labels/train/*.txt → {ruta_imagen_absoluta: set(class_ids)}.

    Solo cuenta imágenes que existen en images_dir (empareja por stem).
    """
    by_stem_img: "dict[str, Path]" = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            by_stem_img[p.stem] = p

    image_classes: "dict[str, set[int]]" = {}
    for lbl in sorted(labels_dir.glob("*.txt")):
        img = by_stem_img.get(lbl.stem)
        if img is None:
            continue
        classes: "set[int]" = set()
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if parts:
                try:
                    classes.add(int(float(parts[0])))
                except ValueError:
                    continue
        image_classes[str(img.resolve())] = classes
    return image_classes


def build_rfs_list(
    image_classes: "dict[str, set[int]]", t: float, seed: int,
    class_names: "Optional[dict[int, str]]" = None,
) -> "tuple[list[str], dict]":
    """Devuelve (lista_de_rutas_repetidas, auditoría)."""
    freqs = class_image_frequencies(image_classes)
    rc = repeat_factor_per_class(freqs, t)
    ri = repeat_factor_per_image(image_classes, rc)

    rng = random.Random(seed)
    lines: "list[str]" = []
    repeats_by_class: Counter = Counter()  # nº de copias atribuibles a cada clase rara
    for img, r in sorted(ri.items()):
        n = max(1, stochastic_count(r, rng.random()))
        lines.extend([img] * n)
        if n > 1:
            for c in image_classes[img]:
                repeats_by_class[c] += n - 1
    rng.shuffle(lines)

    def _name(c: int) -> str:
        return class_names.get(c, str(c)) if class_names else str(c)

    audit = {
        "t": t,
        "seed": seed,
        "n_images": len(image_classes),
        "n_lines": len(lines),
        "oversample_ratio": round(len(lines) / (len(image_classes) or 1), 3),
        "per_class": {
            _name(c): {
                "image_freq": round(freqs.get(c, 0.0), 5),
                "repeat_factor": round(rc.get(c, 1.0), 3),
                "extra_copies": repeats_by_class.get(c, 0),
            }
            for c in sorted(freqs)
        },
    }
    return lines, audit


def write_dataset_yaml(base_yaml: Path, out_yaml: Path, final_dir: Path, train_list: Path) -> bool:
    """Escribe un dataset.yaml gemelo del base con `train` apuntando a la lista RFS.

    Usa rutas ABSOLUTAS (path + train) para que sea portable sin reescrituras.
    Devuelve False si falta el base o PyYAML.
    """
    if not base_yaml.exists():
        return False
    try:
        import yaml  # type: ignore
    except ImportError:
        return False
    cfg = yaml.safe_load(base_yaml.read_text())
    cfg["path"] = str(final_dir.resolve())
    cfg["train"] = str(train_list.resolve())   # absoluta → no depende de `path`
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return True


def main():
    p = argparse.ArgumentParser(description="Genera train_rfs.txt (Repeat-Factor Sampling)")
    p.add_argument("--final-dir", type=Path, required=True,
                   help="Dataset YOLO (contiene images/train y labels/train)")
    p.add_argument("--t", type=float, default=0.001,
                   help="Umbral de frecuencia RFS (default: 0.001, el de LVIS). OJO: con 4 "
                        "clases NO long-tail, t=0.001 < frecuencias → 0 oversampling (no-op). "
                        "Para que actúe, sube t hacia/por encima de la freq de la clase rara "
                        "(t≈4·freq da ~2x). El script avisa si no tuvo efecto.")
    p.add_argument("--seed", type=int, default=0, help="Semilla del redondeo estocástico")
    p.add_argument("--output", type=Path, default=None,
                   help="Ruta del .txt (default: <final-dir>/train_rfs.txt)")
    p.add_argument("--base-yaml", type=Path, default=PROJECT_ROOT / "configs" / "dataset_vehide4.yaml",
                   help="dataset.yaml base del que copiar val/test/names para el gemelo RFS")
    args = p.parse_args()

    images_dir = args.final_dir / "images" / "train"
    labels_dir = args.final_dir / "labels" / "train"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        log.error("Faltan %s o %s", images_dir, labels_dir)
        sys.exit(1)

    image_classes = read_image_classes(labels_dir, images_dir)
    if not image_classes:
        log.error("No se encontraron pares imagen/label en %s", args.final_dir)
        sys.exit(1)

    lines, audit = build_rfs_list(image_classes, args.t, args.seed)

    out = args.output or (args.final_dir / "train_rfs.txt")
    out.write_text("\n".join(lines) + "\n")
    (args.final_dir / "rfs_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    log.info("✅ RFS: %d imágenes → %d líneas (x%.2f) · t=%s",
             audit["n_images"], audit["n_lines"], audit["oversample_ratio"], args.t)
    for name, m in audit["per_class"].items():
        log.info("   %-14s freq=%.4f  r=%.2f  +%d copias", name,
                 m["image_freq"], m["repeat_factor"], m["extra_copies"])
    log.info("   Lista → %s", out)

    # Aviso si t quedó por debajo de las frecuencias → RFS no hizo nada.
    if audit["oversample_ratio"] < 1.02 and audit["per_class"]:
        rarest_name, rarest = min(audit["per_class"].items(), key=lambda kv: kv[1]["image_freq"])
        f = rarest["image_freq"]
        hint = f" Para ~2x la clase más rara ({rarest_name}, freq={f:.3f}) usa --t {round(4 * f, 3)}." if f > 0 else ""
        log.warning("⚠ RFS SIN EFECTO (x%.2f): t=%s está por debajo de las frecuencias de "
                    "clase. VehiDE-4 no es long-tail como LVIS; sube --t.%s",
                    audit["oversample_ratio"], args.t, hint)

    # dataset.yaml gemelo listo para usar (rutas absolutas)
    rfs_yaml = args.final_dir / "dataset_rfs.yaml"
    if write_dataset_yaml(args.base_yaml, rfs_yaml, args.final_dir, out):
        log.info("   Dataset RFS → %s", rfs_yaml)
        log.info("   Entrena con:  python scripts/train.py --data %s ...", rfs_yaml)
    else:
        log.info("   (sin base-yaml/PyYAML; en tu dataset.yaml pon  train: %s)", out.name)


if __name__ == "__main__":
    main()
