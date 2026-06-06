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
- [x] Crear `scripts/audit_log.py`.
- [x] Cada inferencia genera línea JSONL en `logs/inference_{YYYYMMDD}.jsonl` con: timestamp, input_hash (SHA256 de la imagen), model_version, output_summary (lane + total_eur + n_damages), rule_id_applied, processing_time_ms.
- [x] Rotación diaria. No PII en los logs (sin matrícula, sin nombre).
- **Tests**: una inferencia genera exactamente una línea JSONL parseable; hash es determinista.
      ✓ 2026-06-06 · `scripts/audit_log.py`: `hash_image`/`hash_bytes` (SHA256 determinista), `build_record` puro y **PII-safe por diseño** (solo claves whitelisted, sin vía para metadata libre), `log_inference` (1 línea JSONL/eval) y `log_from_output` (extrae del output validado de T1.2 — lo usará T1.5). Rotación diaria por fecha DERIVADA del timestamp del registro (testeable). Config en `configs/audit_log.yaml` (`log_dir`+`filename_pattern`, sin rutas hardcoded). `tests/test_audit_log.py`: 8 tests (hash determinista==hashlib, archivo inexistente, 1 línea parseable, append mismo día, rotación entre días, claves⊆whitelist, sin tokens PII, mapeo de `log_from_output`), todos verdes; cobertura módulo 94%. Suite completa 55/55.

### T1.5 — Orquestador principal (`assess_claim.py`)
- [x] Crear `scripts/assess_claim.py` como punto de entrada operativo único.
- [x] Flujo: cargar imágenes → quality gate por imagen → inferencia daños + zonas → agregación (placeholder hasta T2.5) → triaje → audit log → emitir JSON validado.
- [x] CLI: `python scripts/assess_claim.py --claim-id X --images dir/ --metadata file.json`.
- **Tests**: ejecuta sobre 3 imágenes mock + metadata mock y produce JSON válido contra el schema; deja entrada en audit log.
- **Criterio de aceptación del Sprint 1**: `python scripts/assess_claim.py` corre end-to-end sobre un caso sintético y produce output válido + log. Ningún test falla.
      ✓ 2026-06-06 · `scripts/assess_claim.py` encadena hash→quality_gate→inferencia(daños solo en imágenes válidas)→agregación(placeholder T2.5)→estimación(placeholder T2.2, ceros+conf 0 → nunca verde)→triaje(T1.3)→`build_output`(T1.2)→audit JSONL(T1.4). Componentes **inyectables** (damage_detector, vehicle_detector, estimator) → pipeline end-to-end **offline**; defaults de producción cargan modelos reales y **fallan con mensaje claro si falta `best.pt`** (verificado por CLI: "fetch best.pt from the remote GPU run (T0.1)"). `next_action` por carril en `lane_rules.yaml` v1.1.0 (no hardcoded). `tests/test_assess_claim.py`: 9 tests (3 imágenes→output válido+1 línea audit; sin imágenes válidas→ámbar; lista vacía→error; estructural→rojo ROJO-1; input_hashes==hash de ficheros; detector sin pesos→FileNotFoundError; find_images), todos verdes; cobertura módulo 73% (resto = CLI + carga real YOLO). Suite completa 64/64.

> **SPRINT 1 CERRADO** (2026-06-06): T1.1–T1.5 ✅. Capa de orquestación operativa funcionando end-to-end sobre caso sintético, con triaje determinista, salida validada por schema y auditoría JSONL. Pendiente real: ejecutar con `best.pt` cuando llegue de la GPU remota (T0.1 ⏸) y sustituir placeholders de coste/severidad/agregación en Sprint 2.

---

## SPRINT 2 — Estimación económica y agregación multi-vista (5-7 días)

Aquí el sistema pasa de "detecta daños" a "estima coste". Es lo que justifica el ROI.

### T2.1 — Tablas de referencia (baremos, precios, piezas)
- [x] Crear `configs/baremo_horas.yaml`: para cada `(parte, tipo_daño, severidad_visual)` → `(horas_chapa, horas_pintura, decision: repair|replace|paint_only)`.
- [x] Crear `configs/precios_taller.yaml`: €/h por provincia (datos placeholder con TODO claro, referencia CETRAA/Centro Zaragoza).
- [x] Crear `configs/piezas.yaml`: 20 piezas top siniestrables × marcas top (Seat, Renault, Peugeot, VW, Toyota, Ford) con precio OEM y aftermarket. Placeholder con TODO.
- [x] Documentar en `configs/REFERENCES.md` qué fuente alimenta cada tabla y proceso de actualización.
- **Tests**: las tres tablas parsean sin errores; cobertura mínima documentada.
      ✓ 2026-06-06 · Las 3 tablas ya existían (61ae209, `0.1.0-PLACEHOLDER`); se mantienen sin tocar. Nuevo `configs/REFERENCES.md`: fuente de cada tabla (CETRAA/Centro Zaragoza; objetivo productivo Audatex/GT Motive/DAT Iberia), mapeos/huecos (`broken`↔`broken_light`, `glass` fuera de alcance), cobertura mínima garantizada y proceso de actualización/gobernanza (versionado + data_lineage + model card). `tests/test_reference_tables.py`: 14 tests — parseo+versión, **marcador PLACEHOLDER presente** (regla 19), estructura/valores (horas≥0, decisiones, pintura€/h≥chapa, IVA 21%, año_rango coherente, oem>0/aftermarket null|>0), **cobertura mínima** (6 marcas, ≥20 piezas, provincias clave + default, fallback núcleo) y **consistencia con el schema** (`part_category`/`repair_decision` ⊆ enums de `inference_output_v1.json`). Todos verdes. Suite completa 78/78.

> **SPRINT 2 iniciado** (2026-06-06): T2.1 ✅. Tablas validadas y documentadas — base lista para T2.2 (estimación de coste).

### T2.2 — Módulo de estimación de coste
- [x] Crear `scripts/estimate_cost.py`.
- [x] Función `estimate_repair_cost(damages, vehicle_metadata, province) -> {total_eur, p25, p75, breakdown, confidence}`.
- [x] Maneja sustitución vs reparación según baremo.
- [x] Devuelve **rango P25-P75**, no solo punto medio. Ante incertidumbre, usar P75 para liquidar (regla conservadora).
- [x] Si una pieza no está en `piezas.yaml`, devolver `confidence: low` y derivar a ámbar.
- **Tests**: caso "rayón paragolpes Seat Ibiza" → coste en rango razonable; caso pieza desconocida → confidence low.
      ✓ 2026-06-06 · `scripts/estimate_cost.py`: `estimate_repair_cost(damages, vehicle_metadata, province)` → `{total_eur, p25_eur, p75_eur, breakdown{mano_obra,piezas,materiales,iva}, confidence(0-1), confidence_label, currency, iva_included, province_used, parts_lookup_missing}` (forma lista para `estimacion` del schema). Fórmula del baremo (horas·tarifa_provincia + materiales 15% pintura + pieza si replace + IVA 21%). OEM/aftermarket por política (edad≤3, valor>30k, faro tech). Rango P25–P75 **heurístico** (banda ±20% mano de obra + spread OEM↔aftermarket), honesto: NO es percentil calibrado → recalibrar en Sprint 3 vs importe pagado. Pieza no catalogada → `fallback_prices` + `parts_lookup_missing` + confidence **low** (0.40) → ámbar. Config en `configs/estimation.yaml` (banda, umbrales OEM, niveles confianza, año ref — sin magic numbers). **Caveat**: modificadores de `precios_taller` (taller concertado / premium / urgencia) NO se auto-aplican en v1; el buffer conservador es P75. `tests/test_estimate_cost.py`: 10 tests (Seat Ibiza scratch razonable, replace aftermarket/OEM por edad y valor, pieza desconocida→low, fallback baremo→medium, orden P25≤total≤P75, provincia default, vacío→0, contrato), todos verdes; cobertura módulo 95%. Suite completa 88/88.

### T2.3 — Matriz de severidad económica
- [x] Eliminar la lógica naïve de severidad por % área de `predict.py` (sustituir, no borrar el archivo).
- [x] Crear `business_rules/severity_matrix.yaml`: cada combinación `(parte_categoria, tipo_daño, extension)` → severidad económica (`leve | moderado | severo`).
- [x] Función `compute_severity(damage_with_zone, cost_estimate)` en `scripts/severity.py`.
- [x] Severidad final = max(severidad_visual, severidad_económica).
- **Tests**: faro xenón con crack pequeño → severo; rayón grande en parachoques plástico → leve a moderado.
      ✓ 2026-06-06 · `severity_matrix.yaml` ya existía (matriz + part_to_category + escalation_rules); se mantiene y se le añaden `cost_severity_thresholds` y `preliminary_visual_thresholds` (bump a v1.1.0). `scripts/severity.py`: `compute_severity(damage, cost_estimate)` = max(severidad_visual=matriz por part/type/extension, severidad_económica=por coste €) + escaladas deterministas: **ESC-3** (faro tech xenón/led/matrix→severo) y **chapa+crack→severo + structural_suspicion** (alimenta triaje ROJO-1); no catalogado→`catalogued=False` + default moderado. `predict.py`: bloque naïve de severidad (magic numbers 2/10) **sustituido** por `severity.preliminary_visual_severity()` (umbrales en YAML, marcado **preliminar/visual**, contrato `summary.severity` intacto, archivo no borrado). ESC-1 (multi-daño) es de nivel siniestro → se aplicará en agregación (T2.5). `tests/test_severity.py`: 10 tests (faro crack→severo, rayón grande plástico→leve/moderado, coste alto eleva, chapa crack→severo+structural, ESC-3 en faro LED, structural nunca leve, part_to_category, no catalogado→moderado, contrato, flag preliminar), todos verdes; cobertura módulo 95%. Suite completa 98/98.

### T2.4 — Detección de alertas (preexistente, fraude, inconsistencia)
- [x] Crear `scripts/alerts.py`.
- [x] Implementar detectores iniciales (heurísticos, no ML):
  - `alert_preexisting_damage`: crop del daño + clasificador simple sobre presencia de óxido/suciedad/decoloración (puede ser placeholder con TODO para v2 con clasificador entrenado).
  - `alert_part_declaration_mismatch`: compara partes detectadas vs `descripcion_asegurado` del metadata (NLP simple, fuzzy match).
  - `alert_multiple_unrelated_damages`: si hay daños en 3+ zonas no contiguas con tipologías muy distintas, marcar.
  - `alert_image_manipulation`: comprobar consistencia de metadata EXIF, presencia de doble compresión JPEG (placeholder con TODO).
- [x] Cada alerta tiene `id`, `severity` (info|warning|critical), `description`.
- [x] Las alertas `critical` fuerzan carril rojo.
- **Tests**: caso con descripción "paragolpes" y daño detectado en puerta → alerta mismatch.
      ✓ 2026-06-06 · `scripts/alerts.py`: 4 detectores heurísticos (sin ML), cada alerta `{id, severity, description, evidence}` conforme al schema. `preexisting` = fracción HSV óxido en el crop (TODO v2 clasificador T4.4); `part_declaration_mismatch` = NLP simple normalizando acentos + mapa término→pieza (config) vs detectadas; `multiple_unrelated_damages` = ≥3 zonas/≥2 tipos (aterriza ESC-1 a nivel siniestro); `image_manipulation` = placeholder deshabilitado (id reservado para ROJO-6). **Política v1**: heurísticas emiten `warning` (→ ámbar), nunca `critical`, para no forzar rojos por falsos positivos; `critical` reservado a manipulación/fraude explícito. Config en `configs/alerts.yaml`. `tests/test_alerts.py`: 13 tests (mismatch del plan paragolpes↔puerta, sin/ con coincidencia, acento-insensible, multiple_unrelated, preexistente óxido/limpio/sin-crop, manipulación placeholder, conformidad con el schema de alerts, **critical→rojo ROJO-3** integración, v1 no emite critical), todos verdes; cobertura módulo 87%. Suite completa 111/111.

### T2.5 — Agregación multi-vista por siniestro
- [x] Crear `scripts/claim_aggregator.py`.
- [x] Función `aggregate_claim(reports_per_image) -> consolidated_report`.
- [x] Deduplicar daños: si dos detecciones en imágenes distintas refieren a la misma `(zona, tipo, área_overlap)`, fusionar.
- [x] Confianza por daño consolidado = media ponderada por confianza individual.
- [x] Coste total = suma sobre daños únicos consolidados, no sobre detecciones.
- [x] Resolver conflictos de zona por voting con peso por confianza.
- **Tests**: 3 fotos del mismo paragolpes con el mismo daño → 1 daño consolidado, no 3.
      ✓ 2026-06-06 · `scripts/claim_aggregator.py`: `aggregate_claim(reports_per_image)` agrupa detecciones por `(tipo, región=part|zone)` y fusiona las de imágenes distintas → daño consolidado único. Confianza = media ponderada por confianza (Σc²/Σc); zona/pieza/categoría = voting ponderado por confianza (resuelve conflictos); extensión/severidad = máx; `structural_suspicion` = OR; `supporting_images` = hashes únicos. **Honestidad (regla 18)**: el "área_overlap" cruzado entre vistas NO es computable (distinto punto de vista) → asocio por `(tipo, pieza/zona)`, no IoU cruzado; YOLO ya hace NMS intra-imagen; dos daños distintos del mismo tipo en la misma pieza se reportan como 1 en v1 (limitación documentada). El coste se calculará sobre los consolidados en T2.6. `tests/test_claim_aggregator.py`: 9 tests (3 fotos paragolpes→1, daños distintos→2, media ponderada, voting de zona, OR estructural, extensión/severidad máx, fallback por zona, vacío), todos verdes; cobertura módulo 97%. Suite completa 120/120.

### T2.6 — Integrar todo en `assess_claim.py`
- [x] Actualizar el orquestador para incluir: estimación de coste, severidad económica, alertas, agregación.
- [x] Actualizar las reglas de triaje en `lane_rules.yaml` para usar el coste real (no placeholder).
- [x] Actualizar el schema `inference_output_v1.json` si hay campos nuevos → si los hay, crear `v2.json` y mantener `v1.json` (versionado, no rotura).
- **Criterio de aceptación del Sprint 2**: una llamada al orquestador con 4 fotos de un siniestro real produce un JSON con coste estimado en €, rango P25-P75, severidad, alertas y carril asignado. Audit log completo.
      ✓ 2026-06-06 · `assess_claim.py` (v0.2.0) sustituye placeholders por módulos reales: agregación (claim_aggregator) → coste por daño + de siniestro sobre únicos (estimate_cost, tablas cargadas 1 vez) → severidad económica por daño (compute_severity, propaga structural_suspicion) → alertas (detect_alerts; crops no cableados → preexistente se omite, anotado) → triaje con coste/severidad/alertas reales. `audit.rules_versions` registra lane_rules+severity_matrix+baremo/precios/piezas. **No hizo falta v2 del schema**: los campos encajan en v1 (objetos anidados admiten extras; solo la raíz es estricta) — verificado con `build_output`/`validate_output`. **Demo**: 4 fotos golpe paragolpes Seat Ibiza (Zaragoza) → 1 daño, 426.52€ (P25 384.78/P75 637.67), pieza 180€ aftermarket, severidad moderado, carril verde, audit completo.
      ↻ **Regla AMBAR-2 añadida** (aprobada por el usuario): calidad válida sin daño detectado → ámbar (posible falso negativo, nunca verde). `lane_rules.yaml` → v1.2.0 (VERDE-1 exige ≥1 daño; nueva `no_damage_amber`/AMBAR-2 evaluada antes de verde). Implementada en `triage.py`.
      `tests/test_assess_claim.py` reescrito (12 tests: aceptación 4 fotos, agregación 3→1, caso limpio→verde, mismatch→ámbar, pieza replace priceada, estructural→rojo, **sin-daño→AMBAR-2**, sin imágenes válidas→ámbar, vacío→error, hashes, detector sin pesos, find_images). `test_triage.py` +2 (AMBAR-2). Suite completa 125/125.

> **SPRINT 2 CERRADO** (2026-06-06): T2.1–T2.6 ✅. El sistema pasa de "detecta daños" a "estima coste en € y asigna carril" end-to-end (con inferencia inyectada). Pendiente real: `best.pt` de GPU remota (T0.1 ⏸) para correr con el modelo; sustituir datos PLACEHOLDER de las tablas económicas por reales antes de uso productivo.

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