#!/usr/bin/env python3
"""
prepare_for_review.py — Prepara datos auto-etiquetados para revisión humana en CVAT.

Genera un paquete importable en CVAT con las anotaciones pre-cargadas,
priorizado por confianza (las imágenes más dudosas primero).

Uso:
  python scripts/prepare_for_review.py --input data/auto_labeled --output data/review_package
"""

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("prepare_for_review")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {0: "dent", 1: "scratch", 2: "crack", 3: "broken_light"}
CLASS_COLORS = {
    "dent": "#FF4444",
    "scratch": "#FFD700",
    "crack": "#4488FF",
    "broken_light": "#AA44FF",
}


def load_auto_labeled(input_dir: Path) -> tuple[dict, dict]:
    """Carga anotaciones auto-generadas y scores de confianza."""
    coco_file = input_dir / "annotations_auto.json"
    scores_file = input_dir / "confidence_scores.json"

    if not coco_file.exists():
        log.error("No se encuentra %s", coco_file)
        sys.exit(1)

    with open(coco_file) as f:
        coco_data = json.load(f)

    confidence_scores = {}
    if scores_file.exists():
        with open(scores_file) as f:
            confidence_scores = json.load(f)

    return coco_data, confidence_scores


def sort_by_confidence(
    coco_data: dict, confidence_scores: dict
) -> list[dict]:
    """Ordena imágenes por confianza media (ascendente = más dudosas primero)."""
    images = coco_data["images"]

    def get_confidence(img):
        fname = img.get("file_name", "")
        score_info = confidence_scores.get(fname, {})
        return score_info.get("mean_confidence", 0.0)

    return sorted(images, key=get_confidence)


def generate_cvat_xml(
    coco_data: dict,
    sorted_images: list[dict],
    task_name: str = "Vehicle_Damage_Review",
) -> str:
    """Genera XML en formato CVAT 1.1 para importación."""
    root = Element("annotations")

    # Versión
    version = SubElement(root, "version")
    version.text = "1.1"

    # Meta
    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "name").text = task_name
    SubElement(task, "size").text = str(len(sorted_images))
    SubElement(task, "mode").text = "annotation"

    labels_el = SubElement(task, "labels")
    for class_name, color in CLASS_COLORS.items():
        label = SubElement(labels_el, "label")
        SubElement(label, "name").text = class_name
        SubElement(label, "color").text = color
        SubElement(label, "type").text = "polygon"

    # Indexar anotaciones por image_id
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Imágenes y anotaciones
    for idx, img_info in enumerate(sorted_images):
        image_el = SubElement(root, "image")
        image_el.set("id", str(idx))
        image_el.set("name", img_info["file_name"])
        image_el.set("width", str(img_info.get("width", 0)))
        image_el.set("height", str(img_info.get("height", 0)))

        img_anns = anns_by_image.get(img_info["id"], [])
        for ann in img_anns:
            cat_id = ann["category_id"]
            class_name = CLASS_NAMES.get(cat_id, "unknown")

            segmentation = ann.get("segmentation", [])
            if not segmentation or not isinstance(segmentation[0], list):
                continue

            # Convertir polígono COCO a formato CVAT (x1,y1;x2,y2;...)
            polygon = segmentation[0]
            points = []
            for i in range(0, len(polygon), 2):
                points.append(f"{polygon[i]:.1f},{polygon[i+1]:.1f}")

            if points:
                poly_el = SubElement(image_el, "polygon")
                poly_el.set("label", class_name)
                poly_el.set("points", ";".join(points))
                poly_el.set("occluded", "0")

                # Añadir confianza como atributo
                confidence = ann.get("confidence", 0.0)
                attr = SubElement(poly_el, "attribute")
                attr.set("name", "confidence")
                attr.text = f"{confidence:.3f}"

    # Formatear XML
    xml_str = tostring(root, encoding="unicode")
    parsed = minidom.parseString(xml_str)
    return parsed.toprettyxml(indent="  ")


def generate_review_html(
    sorted_images: list[dict],
    confidence_scores: dict,
    coco_data: dict,
    images_dir: Path,
    output_path: Path,
):
    """Genera un dashboard HTML para revisión rápida."""
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    html_parts = ["""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Revisión de Auto-Etiquetado</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        h1 { text-align: center; margin: 20px 0; color: #00d4ff; }
        .stats { display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }
        .stat-card { background: #16213e; padding: 15px 25px; border-radius: 10px; text-align: center; }
        .stat-card .value { font-size: 2em; font-weight: bold; color: #00d4ff; }
        .stat-card .label { font-size: 0.9em; color: #888; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
        .card { background: #16213e; border-radius: 10px; overflow: hidden; border: 2px solid transparent; transition: border-color 0.3s; }
        .card:hover { border-color: #00d4ff; }
        .card img { width: 100%; height: 200px; object-fit: cover; cursor: pointer; }
        .card-info { padding: 10px; }
        .card-info .filename { font-size: 0.85em; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .confidence { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
        .conf-low { background: #ff4444; color: white; }
        .conf-med { background: #ffa500; color: white; }
        .conf-high { background: #44ff44; color: black; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin: 2px; }
        .legend { display: flex; gap: 15px; justify-content: center; margin: 10px 0; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 16px; height: 16px; border-radius: 3px; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 100; justify-content: center; align-items: center; }
        .modal.active { display: flex; }
        .modal img { max-width: 90%; max-height: 90%; object-fit: contain; }
    </style>
</head>
<body>
    <h1>🔍 Dashboard de Revisión — Auto-Etiquetado</h1>
    <p style="text-align:center;color:#888;">Ordenado por confianza (más dudosas primero). Click en imagen para ampliar.</p>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#FF4444"></div> dent</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFD700"></div> scratch</div>
        <div class="legend-item"><div class="legend-color" style="background:#4488FF"></div> crack</div>
        <div class="legend-item"><div class="legend-color" style="background:#AA44FF"></div> broken_light</div>
    </div>
"""]

    # Stats
    total_images = len(sorted_images)
    total_anns = len(coco_data["annotations"])
    all_confs = [s["mean_confidence"] for s in confidence_scores.values() if s["mean_confidence"] > 0]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0

    html_parts.append(f"""
    <div class="stats">
        <div class="stat-card"><div class="value">{total_images}</div><div class="label">Imágenes</div></div>
        <div class="stat-card"><div class="value">{total_anns}</div><div class="label">Detecciones</div></div>
        <div class="stat-card"><div class="value">{avg_conf:.2f}</div><div class="label">Confianza media</div></div>
    </div>
    <div class="grid">
""")

    for img_info in sorted_images[:200]:  # Limitar a 200 para rendimiento
        fname = img_info["file_name"]
        score_info = confidence_scores.get(fname, {})
        mean_conf = score_info.get("mean_confidence", 0.0)
        num_det = score_info.get("num_detections", 0)

        conf_class = "conf-low" if mean_conf < 0.3 else ("conf-med" if mean_conf < 0.5 else "conf-high")

        # Clases detectadas
        img_anns = anns_by_image.get(img_info["id"], [])
        detected_classes = set()
        for ann in img_anns:
            cname = CLASS_NAMES.get(ann["category_id"], "?")
            detected_classes.add(cname)

        badges_html = ""
        for dc in detected_classes:
            color = CLASS_COLORS.get(dc, "#888")
            badges_html += f'<span class="badge" style="background:{color}">{dc}</span>'

        # Usar visualización si existe, si no la imagen original
        viz_path = f"visualizations/viz_{fname}"
        img_src = viz_path

        html_parts.append(f"""
        <div class="card">
            <img src="{img_src}" onclick="openModal(this.src)" alt="{fname}"
                 onerror="this.src='images/{fname}'">
            <div class="card-info">
                <div class="filename">{fname}</div>
                <span class="confidence {conf_class}">{mean_conf:.2f}</span>
                <span style="color:#888;font-size:0.85em;"> · {num_det} det.</span>
                <div>{badges_html}</div>
            </div>
        </div>
""")

    html_parts.append("""
    </div>
    <div class="modal" id="modal" onclick="this.classList.remove('active')">
        <img id="modal-img" src="">
    </div>
    <script>
        function openModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').classList.add('active');
        }
    </script>
</body>
</html>""")

    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    log.info("Dashboard HTML generado: %s", output_path)


def create_cvat_package(
    xml_content: str,
    images_dir: Path,
    sorted_images: list[dict],
    output_path: Path,
):
    """Crea un ZIP importable como CVAT task backup."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Añadir anotaciones
        zf.writestr("annotations.xml", xml_content)

        # Añadir imágenes
        for img_info in sorted_images:
            fname = img_info["file_name"]
            img_path = images_dir / fname
            if img_path.exists():
                zf.write(img_path, f"data/{fname}")

    log.info("Paquete CVAT creado: %s (%.1f MB)",
             output_path, output_path.stat().st_size / 1024 / 1024)


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara datos auto-etiquetados para revisión en CVAT",
    )
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "auto_labeled",
        help="Directorio con datos auto-etiquetados",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "review_package",
        help="Directorio de salida para el paquete de revisión",
    )
    parser.add_argument(
        "--format", choices=["cvat", "both"], default="cvat",
        help="Formato de exportación (default: cvat)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Preparación para Revisión en CVAT        [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Cargar datos
    coco_data, confidence_scores = load_auto_labeled(args.input)
    images_dir = args.input / "images"

    console.print(f"  Imágenes: {len(coco_data['images'])}")
    console.print(f"  Anotaciones: {len(coco_data['annotations'])}")

    # Ordenar por confianza
    sorted_images = sort_by_confidence(coco_data, confidence_scores)

    # Crear directorio de salida
    args.output.mkdir(parents=True, exist_ok=True)

    # Generar XML CVAT
    console.rule("[bold]Generando paquete CVAT[/]")
    xml_content = generate_cvat_xml(coco_data, sorted_images)

    # Guardar XML suelto
    xml_path = args.output / "annotations.xml"
    with open(xml_path, "w") as f:
        f.write(xml_content)

    # Crear ZIP para CVAT
    zip_path = args.output / "cvat_import.zip"
    create_cvat_package(xml_content, images_dir, sorted_images, zip_path)

    # Generar dashboard HTML
    console.rule("[bold]Generando dashboard de revisión[/]")
    html_path = args.output / "review_dashboard.html"
    generate_review_html(sorted_images, confidence_scores, coco_data, images_dir, html_path)

    # Prioridad de revisión
    table = Table(title="🔍 Prioridad de Revisión (Top 10 más dudosas)")
    table.add_column("Imagen", style="cyan")
    table.add_column("Confianza", justify="right")
    table.add_column("Detecciones", justify="right")

    for img in sorted_images[:10]:
        fname = img["file_name"]
        score = confidence_scores.get(fname, {})
        mean_conf = score.get("mean_confidence", 0.0)
        n_det = score.get("num_detections", 0)
        conf_style = "red" if mean_conf < 0.3 else ("yellow" if mean_conf < 0.5 else "green")
        table.add_row(fname, f"[{conf_style}]{mean_conf:.3f}[/]", str(n_det))

    console.print()
    console.print(table)

    console.print(f"\n[bold green]✅ Paquete de revisión listo[/]")
    console.print(f"   CVAT ZIP:    {zip_path}")
    console.print(f"   Dashboard:   {html_path}")
    console.print(f"\n   Para importar en CVAT:")
    console.print(f"   1. Crea un nuevo Task en CVAT")
    console.print(f"   2. Menu → Actions → Upload annotations → CVAT 1.1")
    console.print(f"   3. Sube {zip_path.name}")
    console.print(f"\n   Después: python scripts/export_reviewed.py\n")


if __name__ == "__main__":
    main()
