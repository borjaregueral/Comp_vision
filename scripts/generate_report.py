#!/usr/bin/env python3
"""
generate_report.py — Genera informes de peritación profesionales en HTML.

Produce un informe HTML autocontenido (imágenes en base64) con el análisis
de daños detectados, apto para flujos de trabajo de seguros.

Uso:
  python scripts/generate_report.py --source image.jpg --model best.pt
  python scripts/generate_report.py --source photos_dir/ --output reports/
"""

import argparse
import base64
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("generate_report")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def image_to_base64(image_path: Path = None, cv2_image: np.ndarray = None) -> str:
    """Convierte una imagen a string base64."""
    if cv2_image is not None:
        _, buffer = cv2.imencode(".jpg", cv2_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode("utf-8")
    elif image_path and image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def generate_html_report(
    report: dict,
    original_image_b64: str,
    annotated_image_b64: str,
    title: str = "Informe de Peritación",
) -> str:
    """Genera el HTML del informe de peritación."""
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    summary = report.get("summary", {})
    damages = report.get("damages", [])
    severity = summary.get("severity", "N/A")

    # Color de severidad
    sev_colors = {
        "Leve": ("#2ecc71", "#e8f8f0"),
        "Moderado": ("#f39c12", "#fef5e7"),
        "Severo": ("#e74c3c", "#fdedec"),
    }
    sev_color, sev_bg = sev_colors.get(severity, ("#95a5a6", "#f2f3f4"))

    # Generar filas de tabla de daños
    damage_rows = ""
    class_colors = {
        "dent": "#FF4444", "scratch": "#FFD700",
        "crack": "#4488FF", "broken_light": "#AA44FF",
    }

    for i, d in enumerate(damages, 1):
        color = class_colors.get(d["class"], "#888")
        conf_color = "#2ecc71" if d["confidence"] > 0.7 else ("#f39c12" if d["confidence"] > 0.4 else "#e74c3c")
        damage_rows += f"""
        <tr>
            <td>{i}</td>
            <td><span class="damage-badge" style="background:{color}">{d['class_es']}</span></td>
            <td><span style="color:{conf_color};font-weight:bold">{d['confidence']:.1%}</span></td>
            <td>{d['area_px']:,} px</td>
            <td>{d['area_pct']:.2f}%</td>
        </tr>"""

    if not damage_rows:
        damage_rows = '<tr><td colspan="5" style="text-align:center;color:#999;">Sin daños detectados</td></tr>'

    # Badges de tipos de daño
    type_badges = ""
    for dtype in summary.get("damage_types", []):
        color = class_colors.get(dtype, "#888")
        type_badges += f'<span class="damage-badge" style="background:{color}">{dtype}</span> '

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — {report_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
        }}

        .report {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 40px;
        }}

        .header h1 {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .header .subtitle {{
            font-size: 0.95em;
            opacity: 0.8;
        }}

        .header-meta {{
            display: flex;
            gap: 30px;
            margin-top: 20px;
            font-size: 0.85em;
        }}

        .header-meta .meta-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .header-meta .meta-label {{
            opacity: 0.6;
        }}

        /* Severity badge */
        .severity-banner {{
            background: {sev_bg};
            border-left: 4px solid {sev_color};
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .severity-badge {{
            background: {sev_color};
            color: white;
            padding: 6px 18px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.95em;
        }}

        /* Content sections */
        .section {{
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }}

        .section h2 {{
            font-size: 1.15em;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Images */
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}

        .image-container {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}

        .image-container img {{
            width: 100%;
            display: block;
        }}

        .image-container .label {{
            padding: 8px 12px;
            font-size: 0.8em;
            font-weight: 600;
            color: #666;
            background: #f8f9fa;
            text-align: center;
        }}

        /* Table */
        .damage-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        .damage-table th {{
            background: #f8f9fa;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #e0e0e0;
        }}

        .damage-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}

        .damage-table tr:hover {{
            background: #f8f9fa;
        }}

        .damage-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            color: white;
            font-size: 0.85em;
            font-weight: 500;
        }}

        /* Stats cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}

        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: #1a1a2e;
        }}

        .stat-card .label {{
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }}

        /* Legend */
        .legend {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }}

        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}

        /* Footer */
        .footer {{
            padding: 20px 40px;
            background: #f8f9fa;
            font-size: 0.75em;
            color: #999;
            text-align: center;
        }}

        @media print {{
            .report {{ box-shadow: none; }}
            .image-grid {{ grid-template-columns: 1fr 1fr; }}
        }}

        @media (max-width: 600px) {{
            .image-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
            .header-meta {{ flex-direction: column; gap: 8px; }}
        }}
    </style>
</head>
<body>
    <div class="report">
        <!-- Header -->
        <div class="header">
            <h1>📋 {title}</h1>
            <div class="subtitle">Análisis automático de daños por visión artificial</div>
            <div class="header-meta">
                <div class="meta-item">
                    <span class="meta-label">ID Informe:</span>
                    <strong>{report_id}</strong>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Fecha:</span>
                    <strong>{timestamp}</strong>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Imagen:</span>
                    <strong>{report.get('image', 'N/A')}</strong>
                </div>
            </div>
        </div>

        <!-- Severity -->
        <div class="severity-banner">
            <div>
                <strong>Evaluación de Severidad</strong>
                <div style="font-size:0.85em;color:#666;margin-top:4px;">
                    Área total afectada: {summary.get('total_damage_area_pct', 0):.2f}%
                </div>
            </div>
            <span class="severity-badge">{severity}</span>
        </div>

        <!-- Summary Stats -->
        <div class="section">
            <h2>📊 Resumen</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="value">{summary.get('total_damages', 0)}</div>
                    <div class="label">Daños detectados</div>
                </div>
                <div class="stat-card">
                    <div class="value">{summary.get('total_damage_area_pct', 0):.1f}%</div>
                    <div class="label">Superficie afectada</div>
                </div>
                <div class="stat-card">
                    <div class="value">{len(summary.get('damage_types', []))}</div>
                    <div class="label">Tipos de daño</div>
                </div>
            </div>
            <div style="margin-top:15px;">
                <strong>Tipos detectados:</strong> {type_badges if type_badges else '<span style="color:#999;">Ninguno</span>'}
            </div>
        </div>

        <!-- Images -->
        <div class="section">
            <h2>📷 Imágenes</h2>
            <div class="image-grid">
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{original_image_b64}" alt="Imagen original">
                    <div class="label">Imagen Original</div>
                </div>
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{annotated_image_b64}" alt="Daños detectados">
                    <div class="label">Daños Detectados</div>
                </div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#FF4444"></div> Abolladura</div>
                <div class="legend-item"><div class="legend-dot" style="background:#FFD700"></div> Arañazo</div>
                <div class="legend-item"><div class="legend-dot" style="background:#4488FF"></div> Grieta</div>
                <div class="legend-item"><div class="legend-dot" style="background:#AA44FF"></div> Faro/Piloto roto</div>
            </div>
        </div>

        <!-- Damage Table -->
        <div class="section">
            <h2>🔍 Detalle de Daños</h2>
            <table class="damage-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Tipo de Daño</th>
                        <th>Confianza</th>
                        <th>Área</th>
                        <th>% Superficie</th>
                    </tr>
                </thead>
                <tbody>
                    {damage_rows}
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Informe generado automáticamente por el sistema de fotoperitación basado en IA.</p>
            <p>Los resultados son orientativos y deben ser validados por un perito profesional.</p>
            <p style="margin-top:8px;">Modelo: YOLOv11-seg · Resolución: {report.get('image_size', ['?','?'])[0]}×{report.get('image_size', ['?','?'])[1]}px · Generado: {timestamp}</p>
        </div>
    </div>
</body>
</html>"""

    return html


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera informes de peritación HTML profesionales",
    )
    parser.add_argument("--source", "--image", dest="source", type=Path, required=True,
                        help="Imagen o directorio (alias: --image)")
    parser.add_argument(
        "--model", type=str,
        default=str(PROJECT_ROOT / "runs" / "damage_seg" / "phase2_finetune" / "weights" / "best.pt"),
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--title", type=str, default="Informe de Peritación")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  Generador de Informes de Peritación      [/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")

    # Importar predict.py para reutilizar funciones
    sys.path.insert(0, str(Path(__file__).parent))
    from predict import run_inference, draw_visualization, find_images

    from ultralytics import YOLO

    if not Path(args.model).exists():
        log.error("Modelo no encontrado: %s", args.model)
        sys.exit(1)

    model = YOLO(args.model)
    images = find_images(args.source)

    if not images:
        log.error("No se encontraron imágenes en: %s", args.source)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    generated = 0

    for img_path in images:
        console.print(f"  Procesando: [cyan]{img_path.name}[/]")

        # Inferencia
        report = run_inference(model, img_path, args.conf, imgsz=args.imgsz)

        # Imágenes
        original_b64 = image_to_base64(image_path=img_path)
        viz = draw_visualization(img_path, report)
        annotated_b64 = image_to_base64(cv2_image=viz) if viz is not None else original_b64

        # Generar HTML
        html = generate_html_report(report, original_b64, annotated_b64, args.title)

        # Guardar
        output_file = args.output / f"informe_{img_path.stem}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        severity = report["summary"].get("severity", "N/A")
        n_damages = report["summary"].get("total_damages", 0)
        console.print(f"    → {n_damages} daños, severidad: {severity}")
        console.print(f"    → [green]{output_file}[/]")
        generated += 1

    console.print(f"\n[bold green]✅ {generated} informes generados en: {args.output}[/]\n")


if __name__ == "__main__":
    main()
