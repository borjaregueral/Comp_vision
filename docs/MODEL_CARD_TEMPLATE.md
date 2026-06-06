# Model Card — `<MODEL_VERSION>`

> **Plantilla.** Copia este fichero a `model_cards/v{X.Y}.md` y rellena todos los campos cada vez que se entrene una versión nueva del modelo. Es obligatorio antes de promover un modelo a producción. Marca las secciones no aplicables con `N/A` y razón, nunca las borres.

---

## 1. Identificación

| Campo | Valor |
|---|---|
| Versión | `vX.Y` |
| Fecha de entrenamiento | `YYYY-MM-DD` |
| Hash SHA256 de `best.pt` | `<hash>` |
| Arquitectura | `YOLOv11m-seg` (o la que aplique) |
| Tarea | Segmentación de instancias de daños en vehículos |
| Responsable técnico | `<nombre>` |
| Aprobación Legal | `<nombre, fecha>` |
| Estado | `experimental` / `piloto` / `producción` / `deprecated` |

## 2. Uso previsto

**Caso de uso permitido**:
- Asistencia a tramitadores y peritos en la evaluación de daños de aparcamiento.
- Importes objetivo: 150-1.500€.
- Decisión final humana en todos los casos del carril ámbar y rojo.
- Liquidación automática del carril verde solo si MAE histórico ≤ 120€ verificado.

**Casos de uso prohibidos**:
- Determinación de fraude sin revisión humana.
- Siniestros con lesiones personales.
- Vehículos con valor declarado > 40.000€.
- Daños por causas naturales (granizo, inundación).
- Tasación para compraventa de vehículos usados (no entrenado para esto).

## 3. Datos de entrenamiento

| Dataset | Versión / Fecha | Imágenes | Anotaciones | Licencia |
|---|---|---|---|---|
| VehiDE | | | | Apache 2.0 |
| CarDD | | | | Research |
| Roboflow SInfo | | | | CC BY 4.0 |
| SYNDCAR | | | | CC BY 4.0 |
| Curado interno | | | | Propietario |

**Splits**: train `70%` / val `20%` / test `10%`, estratificado por clase predominante, seed 42.

**Distribución de clases** (en train):
- `dent`: N (%)
- `scratch`: N (%)
- `crack`: N (%)
- `broken_light`: N (%)

**Sesgos conocidos del dataset**:
- Sobre-representación de daños sobre vehículos de gama media (CarDD).
- Anotaciones VehiDE en vietnamita, traducidas; calidad de etiquetado variable.
- Pocos ejemplos de daños sobre colores oscuros y metalizados.
- No hay imágenes nocturnas ni con lluvia significativa.

## 4. Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Fases | 2 (backbone congelado → fine-tuning completo) |
| Epochs fase 1 | 20 |
| Epochs fase 2 | hasta 280 con early stopping (patience 50) |
| Optimizer | AdamW |
| LR inicial fase 1 | 0.01 |
| LR inicial fase 2 | 0.001 |
| Batch size | 8 (o 16 si A100) |
| Image size | 1024 |
| Augmentaciones | mosaic 1.0, mixup 0.15, copy-paste 0.3, HSV jitter, no flip vertical |
| Hardware | `<GPU>` |
| Tiempo total | `<HH:MM>` |

## 5. Métricas técnicas

### En conjunto de test público (estratificado)

| Métrica | Global | dent | scratch | crack | broken_light |
|---|---|---|---|---|---|
| mAP@50 (box) | | | | | |
| mAP@50 (mask) | | | | | |
| Precision | | | | | |
| Recall | | | | | |
| F1 | | | | | |

### En golden set real (cartera Mutua)

| Métrica | Valor | IC 95% |
|---|---|---|
| MAE estimación coste (€) — carril verde | | |
| MAE estimación coste (€) — carril ámbar | | |
| % casos asignados a verde | | |
| Recall daño visible | | |
| Tasa FN en daño estructural sospechado | | |
| Cohen's weighted kappa (severidad vs perito) | | |
| % estimaciones dentro de ±15% del importe real | | |
| Tiempo medio de procesamiento por siniestro | | |

## 6. Calibración de confianza

| Campo | Valor |
|---|---|
| Calibrador aplicado | `none` / `platt` / `isotonic` |
| Brier score (sin calibrar) | |
| Brier score (calibrado) | |
| Curva de calibración | Ver `eval_business/baseline_vX.Y_calibration.png` |

## 7. Auditoría de sesgo

Métricas desglosadas:

| Dimensión | Tramo | n | mAP@50 | MAE € |
|---|---|---|---|---|
| Marca | SEAT | | | |
| Marca | Renault | | | |
| Marca | VW | | | |
| Color | Claro | | | |
| Color | Oscuro | | | |
| Color | Metalizado | | | |
| Provincia | Madrid | | | |
| Provincia | Otra Tier 1 | | | |
| Provincia | Tier 2-3 | | | |

**Sesgos detectados que requieren mitigación**:
- (Listar; si no hay, escribir "Ninguno detectado en este análisis" y citar limitaciones del análisis).

## 8. Limitaciones conocidas

- (Ejemplo) El modelo no detecta de forma fiable abolladuras en superficies con reflejos especulares fuertes; tasa de falsos positivos sube a `X%` en esos casos.
- (Ejemplo) Daños sobre molduras negras texturizadas (parachoques sin pintar) tienen recall `X%` inferior a la media.
- (Listar todas las limitaciones detectadas durante evaluación)

## 9. Diferencias respecto a la versión anterior

| Cambio | Motivo | Impacto medido |
|---|---|---|
| `<dataset/hiperparametro/augmentación cambiado>` | `<razón>` | `<delta en métrica primaria>` |

**Decisión de promoción**: `<promover / rechazar / mantener como experimental>`. Justificación:

## 10. Reproducibilidad

```bash
# Comando exacto para reproducir este entrenamiento
python scripts/train.py \
    --data configs/dataset.yaml \
    --model yolo11m-seg.pt \
    --imgsz 1024 \
    --batch 8 \
    --epochs-phase1 20 \
    --epochs-phase2 280
```

- Hash del commit del repo en el momento del entrenamiento: `<git hash>`
- Hash del dataset (sha256 sobre `data/final/`): `<hash>`
- Random seed: `42`

## 11. Plan de monitorización post-deployment

- Métricas a vigilar mensualmente: MAE €, recall daño visible, FN estructural, kappa.
- Umbrales de alerta (regresión sobre baseline): caída de mAP > 5pp, MAE € > 150, kappa < 0.65.
- Frecuencia de reentrenamiento prevista: `<trimestral / cuando feedback acumulado > X casos>`.
- Responsable de monitorización: `<nombre>`.

## 12. Compliance

- [ ] Cumple criterios de AI Act para sistemas de alto riesgo (Anexo III, 6.b — sistemas de seguros).
- [ ] Cumple criterios DORA para sistemas ICT críticos.
- [ ] Política de protección de datos personales aplicada (anonimización en ingesta).
- [ ] Plan de rollback documentado en `docs/PILOT_PLAN.md`.
- [ ] Tarjeta firmada por responsable técnico y por Legal.

---

*Plantilla version 1.0 · Fecha plantilla: 2026-06-06*
