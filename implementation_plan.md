# PLAN DE IMPLEMENTACIÓN — Comp_vision → Producción Aseguradora

> **Cómo usar este plan**: ejecuta las tareas en orden. Cada una tiene un objetivo, archivos a tocar, tests requeridos y criterio de aceptación. No saltes sprints. Marca `[x]` al completar y añade fecha + hash de commit en la nota.

---

## SPRINT 0 — Pre-flight check (1 día)

Antes de tocar nada, verifica el estado del entrenamiento y prepara la infraestructura.

### T0.1 — Verificar estado del entrenamiento en curso
- [x] Comprobar que el entrenamiento Fase 2 está corriendo o ha terminado.
- [ ] Si está corriendo: documentar epoch actual, mejores métricas hasta ahora, ETA.
- [ ] Si ha terminado: copiar `best.pt` a `models/baseline_v1.0/` y registrar en `data_lineage.yaml`.
- **Salida**: `models/baseline_v1.0/best.pt` + `models/baseline_v1.0/training_metadata.json`.
      ⏸ EN ESPERA 2026-06-06: el run Fase 2 corre en GPU remota (SSH, ver `setup_gpu.sh`); no hay `best.pt` en local (`runs/` vacío, sin `results.csv`, único `.pt` es el base COCO `yolo11m-seg.pt`). No se puede generar `models/baseline_v1.0/` ni documentar epoch/métricas/ETA hasta tener acceso a los artefactos del run. Confirmado con el usuario.

### T0.2 — Crear estructura de directorios nueva
- [x] Crear `schemas/`, `logs/`, `model_cards/`, `business_rules/`, `tests/`, `golden_set/` (gitignored), `eval_business/`.
- [x] Añadir `.gitkeep` donde proceda.
- [x] Actualizar `.gitignore` con `golden_set/`, `logs/`, `models/*/best.pt`.
      ✓ 2026-06-06 · `schemas/` y `business_rules/` ya existían (commit 61ae209); creados `logs/`, `model_cards/`, `tests/`, `golden_set/`, `eval_business/` con `.gitkeep`. `.gitignore`: `golden_set/*` y `logs/*` (con negación para `.gitkeep`/`README.md`) + `models/*/best.pt` y `models/*/last.pt`. Verificado con `git check-ignore`. (commit f932679)

### T0.3 — Setup de testing
- [x] Añadir `pytest`, `pytest-cov`, `jsonschema` a `requirements.txt`.
- [x] Crear `tests/conftest.py` con fixtures básicas (imagen sintética, predicción mock, metadata mock).
- [x] Configurar `pyproject.toml` o `pytest.ini` para ejecutar con `pytest tests/`.
- **Tests**: `pytest --collect-only` debe descubrir la carpeta sin errores.
      ✓ 2026-06-06 · venv gestionado con `uv` (sin pip): pytest 9.0.3 + pytest-cov 7.1.0 instalados con `uv pip`. `pytest.ini` con `testpaths=tests`. `conftest.py` con fixtures (imagen sintética + variantes borrosa/oscura para T1.1, predicción mock = contrato de `predict.run_inference`, metadata mock = campos de `lane_rules.yaml`). `tests/test_fixtures_smoke.py`: 5 tests, todos verdes. `pytest --collect-only` OK. (commit f932679)

---

## SPRINT 1 — Capa de orquestación operativa (3-5 días)

Aquí transformas el detector en un sistema con triaje. No cambia el modelo, cambia todo a su alrededor.

### T1.1 — Quality Gate (filtro de calidad de imagen)
- [x] Crear `scripts/quality_gate.py`.
- [x] Funciones: `assess_sharpness` (varianza Laplaciano), `assess_exposure` (% píxeles saturados sub/sobre), `assess_resolution` (min 800x600), `detect_vehicle_present` (usar `yolo11n.pt` COCO, clase car/truck), `extract_and_strip_exif`.
- [x] Salida: dict `{valid: bool, problems: list[str], scores: dict, exif_removed: bool}`.
- [x] Configuración en `configs/quality_gate.yaml`: umbrales por criterio.
- **Tests**: imagen borrosa → invalid; imagen oscura → invalid; imagen OK → valid; imagen sin coche → invalid.
- **Criterio**: `pytest tests/test_quality_gate.py` pasa al 100%.
      ✓ 2026-06-06 · commit 47249c5 · `scripts/quality_gate.py` consume `configs/quality_gate.yaml` (umbrales externos, sin magic numbers). `detect_vehicle_present` con YOLO `yolo11n.pt` *lazy* e inyectable; EXIF *stripping* RGPD vía round-trip numpy. `tests/test_quality_gate.py`: 13 tests, cobertura módulo 77%. Fixture `synthetic_image` = checkerboard de alta frecuencia (nitidez=2941≫80).
      ↻ Revisión aplicada 2026-06-06: (#2) resolución orientación-independiente (long/short side) — una vertical 600×800 ya no se rechaza; (#4) `vehicle_area_fraction` por unión de cajas (no suma, ≤1.0) + bbox de unión; (#1) nitidez medida en el ROI del vehículo (`use_vehicle_roi`), con limitación de chapa lisa documentada en YAML/model card. `quality_gate.yaml` → v1.1.0. 16 tests verdes, cobertura 79% (commit 0a56738).

### T1.2 — Schema JSON de salida v1
- [x] Crear `schemas/inference_output_v1.json` (JSON Schema) con campos: `id_evaluacion`, `timestamp`, `version_modelo`, `quality`, `damages[]`, `zones`, `alerts[]`, `estimacion`, `lane`, `lane_reason`, `next_action`, `audit`.
- [x] Crear `scripts/output_builder.py` que tome los outputs intermedios y los emita validados contra el schema.
- [x] Si la validación falla, lanzar excepción explícita (no fallar silenciosamente).
- **Tests**: output válido pasa validación; output con campo faltante falla con mensaje claro.
      ✓ 2026-06-06 · El schema ya existía (commit 61ae209) y cubre todos los campos (nombres en inglés: `model_version`, `zones_summary`); se mantiene como fuente de verdad, sin reescribir. `scripts/output_builder.py`: `build_output()` ensambla el dict canónico (genera `id_evaluacion` `EVA-YYYYMMDD-XXXXXXXX` y `timestamp` UTC RFC3339; lee `schema_version` del propio schema), valida con `jsonschema` Draft 2020-12 (+FormatChecker) y lanza `OutputValidationError` (subclase de `ValueError`) listando cada ruta fallida. `tests/test_output_builder.py`: 10 tests (válido, id/timestamp, campo faltante top-level y anidado, enum `lane`, patrón `lane_rule_id`, `additionalProperties`, skip-validate), todos verdes; cobertura módulo 92%. (commit 8cb2159)

### T1.3 — Triaje determinista (verde/ámbar/rojo)
- [x] Crear `scripts/triage.py` y `business_rules/lane_rules.yaml`.
- [x] Implementar `assign_lane(report, metadata) -> (lane, rule_id, reason)`.
- [x] Reglas en YAML (no en código), con ID estable (`VERDE-1`, `AMBAR-2`, `ROJO-3`) y `effective_date`.
- [x] Reglas iniciales:
  - **ROJO**: daño estructural sospechado / `total_eur > 1500` (placeholder hasta T2.1) / vehículo `valor > 40000` / `siniestros_12m >= 4` / alerta de fraude.
  - **VERDE**: `confidence_mean >= 0.85` Y `quality.valid` Y `total_eur < 800` Y sin alertas Y `siniestros_12m <= 2`.
  - **ÁMBAR**: resto.
- [x] Output incluye `lane`, `rule_id`, `reason_human_readable`.
- **Tests**: 6 casos cubriendo cada regla; un caso "edge" justo en el umbral.
      ✓ 2026-06-06 · `lane_rules.yaml` ya existía (61ae209) y se mantiene como fuente de verdad. **Decisión de diseño**: umbrales/IDs/orden/`reason_template` en YAML; predicados deterministas en `triage.py` enlazados por ID de regla (sin `eval` de las cadenas `condition` del YAML → seguridad + auditabilidad, reglas 10/15). `assign_lane` evalúa ROJO (en orden) → VERDE (todas las condiciones) → AMBAR (resto, con `missing_criteria` en el motivo). `estimacion` ausente se maneja sin romper (→ ámbar). `tests/test_triage.py`: 16 tests (ROJO-1..6 incl. additional_condition de ROJO-6, VERDE-1, AMBAR-1, bordes 800/1500/0.85, precedencia rojo>verde, estimación ausente, contrato lane/rule_id), todos verdes; cobertura módulo 94% (lo no cubierto = ramas defensivas: archivo inexistente, fallbacks de formato, guard de ID desconocido). Suite completa 47/47.

### T1.4 — Logging estructurado de auditoría
- [ ] Crear `scripts/audit_log.py`.
- [ ] Cada inferencia genera línea JSONL en `logs/inference_{YYYYMMDD}.jsonl` con: timestamp, input_hash (SHA256 de la imagen), model_version, output_summary (lane + total_eur + n_damages), rule_id_applied, processing_time_ms.
- [ ] Rotación diaria. No PII en los logs (sin matrícula, sin nombre).
- **Tests**: una inferencia genera exactamente una línea JSONL parseable; hash es determinista.

### T1.5 — Orquestador principal (`assess_claim.py`)
- [ ] Crear `scripts/assess_claim.py` como punto de entrada operativo único.
- [ ] Flujo: cargar imágenes → quality gate por imagen → inferencia daños + zonas → agregación (placeholder hasta T2.5) → triaje → audit log → emitir JSON validado.
- [ ] CLI: `python scripts/assess_claim.py --claim-id X --images dir/ --metadata file.json`.
- **Tests**: ejecuta sobre 3 imágenes mock + metadata mock y produce JSON válido contra el schema; deja entrada en audit log.
- **Criterio de aceptación del Sprint 1**: `python scripts/assess_claim.py` corre end-to-end sobre un caso sintético y produce output válido + log. Ningún test falla.

---

## SPRINT 2 — Estimación económica y agregación multi-vista (5-7 días)

Aquí el sistema pasa de "detecta daños" a "estima coste". Es lo que justifica el ROI.

### T2.1 — Tablas de referencia (baremos, precios, piezas)
- [ ] Crear `configs/baremo_horas.yaml`: para cada `(parte, tipo_daño, severidad_visual)` → `(horas_chapa, horas_pintura, decision: repair|replace|paint_only)`.
- [ ] Crear `configs/precios_taller.yaml`: €/h por provincia (datos placeholder con TODO claro, referencia CETRAA/Centro Zaragoza).
- [ ] Crear `configs/piezas.yaml`: 20 piezas top siniestrables × marcas top (Seat, Renault, Peugeot, VW, Toyota, Ford) con precio OEM y aftermarket. Placeholder con TODO.
- [ ] Documentar en `configs/REFERENCES.md` qué fuente alimenta cada tabla y proceso de actualización.
- **Tests**: las tres tablas parsean sin errores; cobertura mínima documentada.

### T2.2 — Módulo de estimación de coste
- [ ] Crear `scripts/estimate_cost.py`.
- [ ] Función `estimate_repair_cost(damages, vehicle_metadata, province) -> {total_eur, p25, p75, breakdown, confidence}`.
- [ ] Maneja sustitución vs reparación según baremo.
- [ ] Devuelve **rango P25-P75**, no solo punto medio. Ante incertidumbre, usar P75 para liquidar (regla conservadora).
- [ ] Si una pieza no está en `piezas.yaml`, devolver `confidence: low` y derivar a ámbar.
- **Tests**: caso "rayón paragolpes Seat Ibiza" → coste en rango razonable; caso pieza desconocida → confidence low.

### T2.3 — Matriz de severidad económica
- [ ] Eliminar la lógica naïve de severidad por % área de `predict.py` (sustituir, no borrar el archivo).
- [ ] Crear `business_rules/severity_matrix.yaml`: cada combinación `(parte_categoria, tipo_daño, extension)` → severidad económica (`leve | moderado | severo`).
- [ ] Función `compute_severity(damage_with_zone, cost_estimate)` en `scripts/severity.py`.
- [ ] Severidad final = max(severidad_visual, severidad_económica).
- **Tests**: faro xenón con crack pequeño → severo; rayón grande en parachoques plástico → leve a moderado.

### T2.4 — Detección de alertas (preexistente, fraude, inconsistencia)
- [ ] Crear `scripts/alerts.py`.
- [ ] Implementar detectores iniciales (heurísticos, no ML):
  - `alert_preexisting_damage`: crop del daño + clasificador simple sobre presencia de óxido/suciedad/decoloración (puede ser placeholder con TODO para v2 con clasificador entrenado).
  - `alert_part_declaration_mismatch`: compara partes detectadas vs `descripcion_asegurado` del metadata (NLP simple, fuzzy match).
  - `alert_multiple_unrelated_damages`: si hay daños en 3+ zonas no contiguas con tipologías muy distintas, marcar.
  - `alert_image_manipulation`: comprobar consistencia de metadata EXIF, presencia de doble compresión JPEG (placeholder con TODO).
- [ ] Cada alerta tiene `id`, `severity` (info|warning|critical), `description`.
- [ ] Las alertas `critical` fuerzan carril rojo.
- **Tests**: caso con descripción "paragolpes" y daño detectado en puerta → alerta mismatch.

### T2.5 — Agregación multi-vista por siniestro
- [ ] Crear `scripts/claim_aggregator.py`.
- [ ] Función `aggregate_claim(reports_per_image) -> consolidated_report`.
- [ ] Deduplicar daños: si dos detecciones en imágenes distintas refieren a la misma `(zona, tipo, área_overlap)`, fusionar.
- [ ] Confianza por daño consolidado = media ponderada por confianza individual.
- [ ] Coste total = suma sobre daños únicos consolidados, no sobre detecciones.
- [ ] Resolver conflictos de zona por voting con peso por confianza.
- **Tests**: 3 fotos del mismo paragolpes con el mismo daño → 1 daño consolidado, no 3.

### T2.6 — Integrar todo en `assess_claim.py`
- [ ] Actualizar el orquestador para incluir: estimación de coste, severidad económica, alertas, agregación.
- [ ] Actualizar las reglas de triaje en `lane_rules.yaml` para usar el coste real (no placeholder).
- [ ] Actualizar el schema `inference_output_v1.json` si hay campos nuevos → si los hay, crear `v2.json` y mantener `v1.json` (versionado, no rotura).
- **Criterio de aceptación del Sprint 2**: una llamada al orquestador con 4 fotos de un siniestro real produce un JSON con coste estimado en €, rango P25-P75, severidad, alertas y carril asignado. Audit log completo.

---

## SPRINT 3 — Golden set y métricas de negocio (5-7 días)

Sin esto, no sabes si el sistema funciona en producción.

### T3.1 — Definición del golden set
- [ ] Documentar en `golden_set/README.md`: criterios de selección (500-1.000 siniestros cerrados de parking, importes <1.500€, último año), fuentes (extracto de cartera Mutua), proceso de anonimización.
- [ ] Definir esquema de ground truth: archivo JSON por siniestro con campos canónicos (importe final pagado, piezas reparadas/sustituidas, horas reales, decisión final del perito, severidad oficial).
- [ ] **Esta tarea es coordinación con Mutua, no código**. Bloquea T3.2.

### T3.2 — Carga y validación del golden set
- [ ] Crear `scripts/load_golden_set.py`.
- [ ] Validar cada entrada contra el esquema definido en T3.1.
- [ ] Estratificar por tramos de importe: `<500`, `500-1500`, `>1500` (este último es de control, debe ir a rojo).
- [ ] Reportar estadísticas: distribución por marca, color, provincia, tipo de daño dominante.
- **Tests**: rechaza entradas con campos faltantes; estratifica correctamente.

### T3.3 — Métricas de negocio
- [ ] Crear `scripts/business_metrics.py`.
- [ ] Implementar:
  - **MAE en €**: por carril (debe medirse solo en verde + ámbar), por tramo de importe.
  - **% casos en carril verde**: sobre el total y sobre los liquidables.
  - **Tasa de FN en daño estructural sospechado**: contra ground truth.
  - **Cohen's weighted kappa**: clasificación de severidad modelo vs perito.
  - **% estimaciones dentro de ±15% del valor real**.
  - **Tiempo medio de procesamiento** por siniestro.
- [ ] Reporte HTML autocontenido en `eval_business/report_{fecha}.html` con tablas, intervalos de confianza (bootstrap), y gráficos.
- **Tests**: sobre un golden set sintético de 20 casos, todas las métricas se calculan sin error y devuelven valores en rangos esperables.

### T3.4 — Calibración de confianza
- [ ] Crear `scripts/calibrate_confidence.py`.
- [ ] Sobre el golden set (o validation si golden no está aún): construir curva de calibración (reliability diagram).
- [ ] Aplicar isotonic regression o Platt scaling sobre las confianzas de salida del modelo.
- [ ] Guardar el calibrador como `models/baseline_v1.0/confidence_calibrator.pkl`.
- [ ] Integrar en `predict.py` (vía opción `--calibrate`) sin reentrenar el modelo.
- **Tests**: confianzas tras calibrar tienen Brier score menor o igual que sin calibrar sobre validation.

### T3.5 — Evaluación de la versión 1.0 contra el golden set
- [ ] Ejecutar `business_metrics.py` sobre el golden set completo con el modelo `baseline_v1.0`.
- [ ] Generar `eval_business/baseline_v1.0_report.html`.
- [ ] Crear `model_cards/v1.0.md` con: dataset de entrenamiento, métricas en validation, métricas en golden set, sesgos detectados (por marca/color/provincia), limitaciones conocidas, fecha.
- **Criterio de aceptación del Sprint 3**: el reporte de negocio existe, contiene las 6 métricas clave con intervalos de confianza, y la model card está firmada con el hash del `best.pt`.

---

## SPRINT 4 — Mejora del modelo basada en evidencia (7-10 días)

Ahora sí, mejoras del modelo, pero guiadas por las métricas de negocio del Sprint 3.

### T4.1 — Augmentaciones específicas de parking
- [ ] Añadir a `scripts/train.py` (vía flag opcional, no romper la rama actual): augmentaciones con `albumentations` o `imgaug`:
  - Simulación de reflejos especulares
  - Simulación de superficie mojada
  - Sombras duras de pilares
  - Motion blur leve (foto a pulso)
- [ ] Configuración en `configs/augmentations_parking.yaml`.
- [ ] El nuevo entrenamiento va a `models/v1.1/`, NO sustituye `baseline_v1.0`.
- **Tests**: las augmentaciones se aplican sin romper el pipeline; visualizar 10 ejemplos.

### T4.2 — Dataset de alta calidad (curado)
- [ ] Crear `scripts/curate_dataset.py`.
- [ ] Generar `data/curated/`: CarDD completo + subset revisado de VehiDE (descartar imágenes con etiquetado pobre, marcadas como "low_quality" manualmente o por heurística).
- [ ] Entrenar `models/v1.2/` solo sobre `data/curated/`.
- [ ] Comparar v1.0 vs v1.1 vs v1.2 sobre el golden set.
- **Criterio**: si v1.2 mejora en métrica primaria (MAE €) sobre v1.0, promoverlo. Si no, documentar por qué y mantener v1.0.

### T4.3 — Clases ampliadas (de 4 a 8-10)
- [ ] Diseñar nuevo `configs/data_config_v2.yaml` con clases ampliadas: `scratch_superficial`, `scratch_profundo`, `dent_pdr`, `dent_chapa`, `paint_chip`, `crack`, `broken_light`, `bumper_misalignment`, `panel_gap`.
- [ ] Esta tarea requiere reanotación parcial. Documentar el proceso en `docs/reannotation_protocol.md`.
- [ ] **Esta es la única tarea del plan que puede llevar 2-4 semanas adicionales**. Decidir con stakeholders si entra en v2.0 o se aplaza.

### T4.4 — Detector de daño preexistente entrenado
- [ ] Crear `scripts/train_preexisting_detector.py`.
- [ ] Pequeño clasificador binario (MobileNet o EfficientNet-B0) sobre crops de daños.
- [ ] Dataset propio (a coordinar con Mutua): 500+ crops etiquetados como "fresco" o "preexistente".
- [ ] Integrar en `alerts.py` reemplazando el placeholder de T2.4.
- **Criterio**: F1 ≥ 0.75 sobre validation.

---

## SPRINT 5 — Compliance y operación continua (en paralelo, ongoing)

Tareas que se hacen y se mantienen permanentemente.

### T5.1 — Loop de active learning
- [ ] Crear `scripts/feedback_collector.py`.
- [ ] Cada corrección de un perito sobre output del modelo se guarda en `data/feedback/` con timestamp, claim_id, output_modelo, decisión_humana.
- [ ] Script mensual `scripts/build_retraining_set.py` que añade el feedback al dataset de reentrenamiento.

### T5.2 — Bias testing
- [ ] Crear `scripts/bias_audit.py`.
- [ ] Métricas desglosadas por: marca, color (claro/oscuro/metálico), provincia, antigüedad del vehículo, sexo del asegurado si está disponible y es legal.
- [ ] Reporte mensual en `eval_business/bias_{YYYYMM}.html`.

### T5.3 — Data lineage
- [ ] Crear y mantener `data_lineage.yaml`: cada dataset, cada modelo, cada métrica reportada al negocio queda con su origen, fecha, hash, persona responsable.

### T5.4 — Documentación final
- [ ] Crear `docs/SYSTEM_OVERVIEW.md`: arquitectura, flujos, decisiones de diseño.
- [ ] Crear `docs/OPERATOR_MANUAL.md`: cómo interpretar un output, cuándo escalar a perito, glosario de alertas.
- [ ] Crear `docs/AUDIT_GUIDE.md`: cómo auditar una decisión histórica del sistema (cruzar logs + model card + datos de entrada).

---

## REGLAS DE TRACKING

Al completar cada tarea, edita el checkbox y añade una línea:

```
- [x] T1.1 — Quality Gate
      ✓ 2026-06-10 · commit a3f4b9c · 12 tests passing · cobertura módulo 87%
```

Si una tarea se bloquea, añade nota:

```
- [ ] T3.1 — Golden set
      ⏸ BLOQUEADO 2026-06-12: esperando confirmación de Legal sobre anonimización
```

## DEFINICIÓN DE HECHO PARA EL PROYECTO ENTERO

El proyecto está "listo para piloto" cuando:
1. Sprint 1 + Sprint 2 + Sprint 3 cerrados.
2. `eval_business/baseline_v1.0_report.html` muestra: MAE € ≤ 150, recall daño visible ≥ 90%, FN estructural ≤ 3%.
3. Model card v1.0 firmada por el responsable técnico y por Legal.
4. Existe un plan operativo escrito en `docs/PILOT_PLAN.md` con: alcance del piloto, criterios de éxito, plan de rollback.

El proyecto está "listo para producción" cuando además:
5. Sprint 4 cerrado con modelo demostrablemente mejor en al menos 2 de las 6 métricas de negocio.
6. Sprint 5 operativo con al menos 2 meses de feedback acumulado y un reentrenamiento ejecutado con éxito.
7. Auditoría externa (interna de Mutua o consultora) firmada conforme a AI Act y DORA.