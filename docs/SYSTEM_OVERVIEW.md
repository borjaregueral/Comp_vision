# Visión General del Sistema — Fotoperitación de Daños en Vehículos

> Documento técnico de referencia: arquitectura, modelos, decisiones de diseño y su
> justificación. Audiencia: revisor técnico. Estado: **experimental** (sin firma de producción).

---

## 1. Qué hace el sistema

Automatiza la **primera línea de evaluación de siniestros de daños de aparcamiento**.
Entrada: fotos del vehículo dañado + metadatos del siniestro. Salida: una evaluación
estructurada, **validada por esquema y auditada** — qué daño, dónde, coste estimado en €,
alertas de fraude/inconsistencia, y una **decisión de triaje** (resolución automática /
revisión humana rápida / peritaje presencial obligatorio).

**Principio de diseño central (decirlo lo primero en la discusión):**
> El modelo de visión **nunca decide nada operativo**. Solo aporta *candidatos de daño*.
> Todas las decisiones de dinero y enrutamiento las toman **reglas de negocio
> deterministas y versionadas**. Esto acota el impacto del error del modelo, mantiene
> cada decisión **auditable**, y significa que una métrica de detección mediocre **no** se
> traduce en un producto mediocre o inseguro.

---

## 2. Arquitectura end-to-end

```
                 ┌──────────────────── por imagen ───────────────────┐
  fotos ───▶ quality gate ──▶ ┌─ modelo de DAÑOS (YOLOv11m-seg, 6cl)─┐
  + metadatos  (¿nítida?       └─ modelo de PARTES (YOLOv11m-seg,23)─┘
                ¿hay coche?         │ localize: daño ∩ parte → zona + pieza
                ¿exposición?        ▼
                EXIF eliminado)  agregación multi-vista (dedup entre fotos)
                                     ▼
            severidad (matriz) ─▶ coste € (baremo/piezas) ─▶ alertas fraude/inconsistencia
                                     ▼
                 triaje determinista  ──▶  🟢 verde / 🟡 ámbar / 🔴 rojo
                                     ▼
                 JSON validado por esquema  +  línea de auditoría JSONL
```

Dos modelos de visión en dos ejes independientes — **qué** es el daño (modelo de daños) y
**dónde** está (modelo de partes) — que alimentan una capa de negocio determinista.
Punto de entrada: [`scripts/assess_claim.py`](../scripts/assess_claim.py).

---

## 3. Los modelos

### 3.1 Modelo de DAÑOS — `v1.2`
- **Arquitectura:** YOLOv11m-seg (Ultralytics) — **segmentación de instancias** de una etapa.
- **Clases (6):** `scratch, dent, crack, paint_chip, puncture, broken_light`.
- **Pesos:** `models/v1.2/best.pt` (sha256 `d56a7968…`). Ficha: [`model_cards/v1.2.md`](../model_cards/v1.2.md).
- **Entrenamiento:** 2 fases desde COCO-preentrenado `yolo11m-seg.pt`, imgsz **1024**, AdamW.
  - Fase 1: 20 épocas, **backbone congelado** (calienta la cabeza nueva, AMP activado).
  - Fase 2: fine-tuning completo, **`--no-amp`**, batch 4, parado en época 65 (meseta).
- **Métricas (test):** box **mAP50 0.339**, mask mAP50 0.307. Por clase:

  | clase | box mAP50 | recall | nota |
  |---|--:|--:|---|
  | broken_light | 0.64 | 0.73 | la más fuerte (objeto rígido) |
  | scratch | 0.36 | 0.44 | gama media sólida |
  | dent | 0.33 | 0.37 | gama media sólida |
  | crack | 0.32 | 0.41 | gama media sólida |
  | puncture | 0.24 | 0.30 | usable |
  | paint_chip | 0.15 | 0.15 | débil (rara + etiquetas ruidosas) |

### 3.2 Modelo de PARTES — `carparts-seg`
- **Arquitectura:** YOLOv11m-seg, **23 clases** (dataset Ultralytics `carparts-seg`): cada
  parte exterior diferenciada por lado (`front_left_door`, `back_right_light`, …).
- **Pesos:** `models/parts_seg/best.pt`. **box mAP50 0.867 / mask 0.883** (las partes son
  objetos grandes y estables → fáciles de segmentar; de ahí la métrica alta vs. el de daños).
- **Función:** segmenta las partes del coche; cada daño se asigna a la **parte con la que más
  solapa** → da una **zona** (front/rear/front_left/…) y una **part_category**
  (body_panel/plastic_panel/light_assembly/…) que se usa para costear. Ver
  [`scripts/localize.py`](../scripts/localize.py).
- **Convención izquierda/derecha — VERIFICADA relativa al vehículo** (2026-06-12):
  `front_left_*` es el lado real izquierdo del coche (conductor, LHD), no de la cámara.
  Comprobado renderizando etiquetas en vistas frontal + trasera (el lado cambia de lado de
  imagen según el ángulo). El mapeo de zonas es correcto — **no hay inversión de lado**.

---

## 4. Decisiones de diseño clave (y su razonamiento)

| # | Decisión | Por qué | Alternativa considerada | Compromiso aceptado |
|---|---|---|---|---|
| 1 | **YOLOv11-seg**, no ResNet | ResNet es un *clasificador/backbone* — no localiza. Necesitamos detectar+segmentar+clasificar cada daño, una pasada, tiempo real, pesos preentrenados fuertes. | **Mask R-CNN** (backbone ResNet) — la comparación real. | YOLO por velocidad/simplicidad/backbone moderno; Mask R-CNN es más pesado/2 etapas. |
| 2 | **Segmentación** (máscaras), no solo cajas | Las máscaras permiten asignar el daño a la pieza por solape; formas irregulares; daños entre dos piezas. | Solo detección (cajas) | Las máscaras son más ruidosas de etiquetar/aprender (limita el mAP), pero habilitan el paso de zona/pieza. |
| 3 | **Dos modelos** (daños + partes), una cascada | El coste depende de la *pieza concreta*; el modelo de daños no la conoce. Un segmentador de partes da zona + pieza. | Modelo único multitarea; heurísticas geométricas | Riesgo de error en cascada (un fallo de partes corrompe el coste) — mitigado con abstención (`zone=unknown`) + enrutado a humano por baja confianza. |
| 4 | **Visión aporta candidatos; reglas deterministas deciden dinero/carril** | Auditabilidad, impacto acotado del error, defensibilidad regulatoria (AI Act/DORA). | Decisión ML extremo a extremo | Menos "inteligente", pero cada € y carril es explicable y versionado. |
| 5 | **Taxonomía de 6 clases** (ni 4, ni 8–10) | 4 era ruidosa/gruesa; 8–10 (`dent_pdr`, `panel_gap`) no es distinguible de forma fiable desde un solo crop. 6 es el punto pragmático. | Mantener 4; ir a 8–10 | `paint_chip`/`puncture` son minoritarias/difíciles, pero cada una mapea a una decisión de reparación distinta. |
| 6 | **Crear granularidad re-etiquetando crops existentes** (CLIP), sin fotos nuevas | La granularidad de sub-tipo no existe en ninguna etiqueta de origen, y no hay fotos nuevas. | Adquirir/anotar datos nuevos | Etiquetas auto-generadas (revisión por muestreo), más ruidosas que humanas. |
| 7 | **La etiqueta gruesa restringe la fina** en el auto-etiquetado | La etiqueta gruesa fiable (CarDD/VehiDE) restringe la elección de CLIP → alta calidad donde la tenemos; riesgo total solo en los 20k boxes de Roboflow sin tipo. | CLIP a ciegas sobre todo | Concentra y acota el ruido. |
| 8 | **Piso de confianza 0.55** sobre las auto-etiquetas | El ruido vive en la banda baja; el piso lo elimina. Imágenes vacías se descartan (sin falso fondo). | Entrenar con todo / verificar humano todo | Pierde algo de recall (menos etiquetas) para ganar pureza. |
| 9 | **`--no-amp` en fase 2** | La precisión mixta (AMP) hace que `seg_loss` diverja a **NaN** al descongelar el backbone. | Mantener AMP | FP32 dobla la VRAM → batch 8→4, más lento. |
| 10 | **imgsz 1024** | Rayones/grietas son finos y pequeños — necesitan resolución. | 640 (más rápido) | Más cómputo. |

---

## 5. Pipeline de datos

**Fuentes:** CarDD, VehiDE (etiquetas en vietnamita), Roboflow. Resultado en `data/final_v2`:
**39.724 anotaciones / 17.498 imágenes**, splits 12.246 / 3.497 / 1.755 (estratificado, semilla 42).

1. `download_datasets.py` → `unify_to_yolo.py`: COCO → YOLO-seg, mapeo de clases.
2. **`auto_relabel.py`** (el paso v2): CLIP zero-shot re-tipa cada crop de daño existente,
   **restringido por su etiqueta gruesa**, y **recupera 20.082 boxes `Damage` de Roboflow**
   antes descartados (localizados pero sin tipo). Emite por anotación
   `fine_conf` / `fine_method` / `needs_review` + un HTML de revisión por muestreo.
3. `unify_to_yolo.py --min-conf 0.55`: aplica el piso, descarta imágenes vacías, escribe
   `data/final_v2` + `configs/dataset_v2.yaml`.

Config de taxonomía: [`configs/taxonomy_v2.yaml`](../configs/taxonomy_v2.yaml). Trazabilidad:
[`data_lineage.yaml`](../data_lineage.yaml).

### 5.1 Alcance y exclusiones (qué NO cubre el sistema)

**Fuera de alcance por decisión explícita:** lunas/cristales (parabrisas, ventanillas) y
neumáticos/pinchazos. Se aplica en **dos niveles** en `configs/data_config.yaml`:

1. **Mapeo explícito a `null`** de las clases conocidas de origen: `Glass Shatter`,
   `Tire Flat`, `vo_kinh` (cristal roto), `mat_bo_phan` (pieza ausente), `flat_tire`,
   `glass_shatter`, `broken_glass`…
2. **Patrones de exclusión por substring** (case-insensitive), como red de seguridad para
   clases no previstas: `tire`, `tyre`, `flat`, `glass`, `windshield`, `window`, `luna`,
   `cristal`, `parabrisas`.

**Matiz v2 (NO confundir con el doc viejo de 4 clases):** en v1 la clase **genérica `Damage`**
(datasets mono-clase tipo Roboflow/SInfo) se mapeaba a `null` — se excluía porque no distingue
tipo y contaminaría las clases. En **v2 ya NO se excluye**: `auto_relabel.py` lee esos
**20.082 boxes `Damage`** directamente del COCO de Roboflow y los **tipa con CLIP**
(recuperación). Las exclusiones de **lunas/neumáticos sí siguen vigentes** en v2 (la taxonomía
de 6 clases no incluye ninguna de esas categorías).

**⚠️ Cuidado con `puncture`:** la clase v2 `puncture` es una **perforación/agujero en un PANEL
de carrocería** (chapa o plástico) — **no** un pinchazo de neumático, que está excluido. Es una
confusión fácil que un revisor podría plantear.

---

## 6. La capa operativa / de negocio (determinista)

| Etapa | Script / config | Qué hace |
|---|---|---|
| Quality gate | `quality_gate.py` | Nitidez (Laplaciano), exposición, resolución, **¿hay vehículo?** (yolo11n COCO), **EXIF eliminado** (RGPD). |
| Agregación | `claim_aggregator.py` | Deduplica el mismo daño entre varias fotos; voto ponderado por confianza para zona/pieza. |
| Severidad | `severity.py` + `severity_matrix.yaml` | `max(severidad visual, severidad económica)`; escaladas (p.ej. grieta/perforación en chapa → estructural → rojo). |
| Coste € | `estimate_cost.py` + baremo/precios/piezas | Horas×tarifa + precio pieza (si replace) + materiales + IVA; **rango P25–P75**; **tablas PLACEHOLDER**. |
| Alertas | `alerts.py` | 4 heurísticas: daño preexistente, mismatch declaración-pieza, múltiples daños no relacionados, manipulación de imagen. |
| Triaje | `triage.py` + `lane_rules.yaml` | **Determinista** verde/ámbar/rojo, reglas en YAML con IDs estables + fechas de efecto. |
| Calibración | `calibrate_confidence.py` | Isotónica/Platt sobre las confianzas (mejora Brier), aplicada en inferencia sin reentrenar. |
| Auditoría | `audit_log.py` | Una línea JSONL por siniestro: hashes de entrada (SHA256), versión del modelo, carril, id de regla, tiempos. **Sin PII.** |
| Salida | `output_builder.py` + `inference_output_v1.json` | Ensambla + **valida contra JSON Schema**; falla ruidosamente si se viola. |

**Carriles de triaje:**
- 🔴 **Rojo** — estructural sospechado / `total > 1500€` / valor vehículo > 40k€ / ≥4 siniestros/año / alerta crítica → peritaje presencial obligatorio.
- 🟢 **Verde** — alta confianza + calidad válida + `< 800€` + sin alertas + ≤2 siniestros → resolución automática.
- 🟡 **Ámbar** — el resto → revisión humana rápida en pantalla.

---

## 7. Cómo leer las métricas

- **mAP50** = precisión media promedio a IoU 0.5 (localización indulgente). **mAP50-95** =
  promediado sobre IoU más estrictos (más difícil). **box** vs **mask** = calidad de caja vs. máscara de píxeles.
- **El recall es la métrica crítica de negocio aquí**, no el mAP. En seguros un **falso
  negativo = daño no detectado = infravaloración**; un falso positivo solo va a ámbar (lo
  mira un humano). Por eso afinamos hacia recall, no precisión.
- **0.339 vs. el 0.434 antiguo:** *no son comparables directamente*. El 0.434 era un modelo de
  **4 clases** con etiquetas ruidosas. v1.2 es de **6 clases** (más difícil) con etiquetas
  limpias y máscaras que funcionan. Sobre las *4 clases compartidas* v1.2 promedia
  **≈0.41 box mAP50** — comparable — añadiendo dos tipos de daño, máscaras y los datos
  recuperados. La media de 6 clases la arrastra hacia abajo las dos minoritarias (sobre todo `paint_chip`).

---

## 8. ⚠️ QUÉ SIGUE SIENDO PLACEHOLDER / INCOMPLETO (inventario completo)

Esto es lo que **NO es real todavía** — decirlo proactivamente transmite rigor.

### A. Económico (lo más importante)
- **`configs/baremo_horas.yaml`** (v0.2.0-**PLACEHOLDER**) — horas de mano de obra orientativas
  (rangos públicos CETRAA/Centro Zaragoza). Las filas de `paint_chip`/`puncture` son placeholder.
- **`configs/precios_taller.yaml`** (0.1.0-**PLACEHOLDER**) — €/hora por provincia.
- **`configs/piezas.yaml`** (0.1.0-**PLACEHOLDER**) — precios de pieza (OEM/aftermarket).
  **Faltan los faros traseros** (`back_left_light`, `back_right_light`) → caen a precio fallback.
- **Rango P25–P75 heurístico** (banda ±20% mano de obra), **no** un percentil calibrado.
- **Modificadores de `precios_taller`** (taller concertado / premium / urgencia) **NO se
  auto-aplican** en v1; el colchón conservador es P75.
- **Extensión (small/medium/large) NO se calcula** relativa a la pieza en el pipeline cableado;
  por defecto "small" en la agregación → el baremo se consulta siempre con extensión "small"
  (afecta a la precisión del coste).
- **Coste = estructuralmente correcto, NO dinero real** hasta cargar el baremo oficial pactado.

### B. Alertas (heurísticas, no ML)
- **`alert_preexisting_damage`** — heurística placeholder (fracción HSV de óxido) **Y los
  crops no están cableados** en `assess_claim` → la alerta de preexistente **se omite** en el
  pipeline actual.
- **`alert_image_manipulation`** — placeholder **deshabilitado** (id reservado para ROJO-6).
- **Detector de preexistente entrenado** (T4.4, clasificador MobileNet/EfficientNet) — **no construido**.

### C. Calibración de confianza
- El calibrador se ajustó con **datos sintéticos** (T3.4). El calibrador **real** sobre
  validación/golden set **no está ajustado** → `--calibrate` usa un ajuste sintético o nada.

### D. Golden set y métricas de negocio (bloqueado por DATOS, no por modelo)
- **Datos reales de cartera de la aseguradora NO obtenidos** (coordinación + anonimización pendientes).
- **Métricas de negocio** (MAE €, % verde, tasa FN estructural, Cohen's κ, % dentro de ±15%)
  solo **probadas sobre golden set SINTÉTICO** → **no hay cifras de negocio reales** (T3.5).
- **Auditoría de sesgo** (T5.2) por marca/color/provincia — **no realizada** (los datasets
  públicos no traen esa metadata; diferido al golden set).

### E. Etiquetas y datos
- Las etiquetas de `data/final_v2` son **CLIP auto-generadas** con **revisión por muestreo**,
  **no verificadas al 100% por humano** (especialmente las `needs_review`).
- Sin **fotos reales de parking** (brecha de dominio) — el dato más importante que falta.

### F. Modelos
- **v1.2 es experimental** — la ficha de modelo está **sin firmar** (técnico + Legal pendientes).
- **`paint_chip` es débil** (recall 0.15) — fusión a 5 clases no realizada.
- **Modelo de partes:** sin **eval por zona sobre fotos reales de parking** (solo métricas de
  carparts-seg).

### G. Otras tareas del plan no hechas / parciales
- `docs/reannotation_protocol.md` (T4.3) — no creado (se fue por auto-relabel).
- Augmentaciones específicas de parking (T4.1), dataset curado (T4.2) — no hechos.
- Loop de active learning (T5.1), documentación final (T5.4 — este doc es parte) — parciales.
- **Agregación multi-vista**: NO deduplica por IoU cruzado entre fotos (puntos de vista
  distintos) → asocia por `(tipo, pieza/zona)`; dos daños del mismo tipo en la misma pieza se
  reportan como 1 (limitación documentada).

---

## 9. Limitaciones conocidas (decirlas proactivamente)

1. **Brecha de dominio** — entrenado con datasets públicos web/accidente, **no** con fotos de
   móvil de parkings reales. Las métricas de test probablemente sobreestiman el rendimiento real. *Lo #1 a arreglar.*
2. **`paint_chip` poco fiable** (recall 0.15) — clase rara + etiquetas más ruidosas.
3. **Etiquetas CLIP auto-generadas** (muestreo), no verificadas 100% por humano.
4. **Economía PLACEHOLDER** — el € es estructuralmente correcto pero no es dinero real.
5. **Sin evaluación en golden set** — métricas de negocio bloqueadas por datos reales de la aseguradora.
6. **Recall modesto** (~0.37–0.44 en clases núcleo) — producción bajaría el umbral y, sobre todo, añadiría datos de dominio.

---

## 10. Preguntas anticipadas — respuestas concisas

**P: ¿Por qué YOLO y no ResNet?**
ResNet es un *clasificador/backbone* — dice "qué hay en la imagen", no localiza ni segmenta
por instancia. Necesitamos detectar + segmentar + clasificar cada daño. La comparación real es
YOLOv11-seg vs. **Mask R-CNN** (que *usa* un backbone ResNet); elegimos YOLO por velocidad (una
etapa), tooling más simple y un backbone moderno más fuerte.

**P: ¿0.339 mAP es bueno?**
Para *segmentación de instancias de daños* está en la banda publicada normal (0.3–0.6) — el daño
tiene bordes difusos y las etiquetas públicas son ruidosas. Y es el número equivocado donde
anclarse: el **recall** manda el resultado de negocio, y `broken_light` (0.64) y las clases
núcleo (~0.33) son sólidas; la media la baja `paint_chip`.

**P: ¿Dos modelos no es frágil?**
El coste depende de la *pieza concreta*, que el modelo de daños no aporta. El riesgo de cascada
(error de partes → precio mal) es real y mitigado: los daños que no solapan una pieza con
confianza quedan `zone=unknown`, el fallback de pieza desconocida da **confianza baja → va a
humano**. Degrada con seguridad.

**P: ¿Cómo sabéis que las etiquetas auto-generadas son correctas?**
No del todo — son CLIP zero-shot con **revisión humana por muestreo**, y aplicamos un **piso de
confianza 0.55** para quitar la cola ruidosa. La etiqueta gruesa restringe la elección fina, así
que el ruido se concentra en el set recuperado de Roboflow, que está acotado.

**P: ¿El coste es ML?**
No. Baremo determinista: `horas × tarifa provincia + precio pieza (si replace) + materiales +
IVA`, devuelto como **rango P25–P75** (conservador → P75 para liquidar). El modelo solo aporta
tipo + pieza. Las tablas son hoy PLACEHOLDER.

**P: ¿Cómo evita auto-pagar un siniestro fraudulento o mal?**
No puede auto-resolver salvo que se cumpla *todo*: alta confianza, calidad válida, coste bajo,
sin alertas, pocos siniestros. Cualquier incertidumbre → ámbar/rojo (humano). Cuatro heurísticas
de fraude/consistencia; las alertas críticas fuerzan rojo.

**P: ¿RGPD / cumplimiento?**
EXIF eliminado en el gate; log de auditoría sin PII (solo hashes); el esquema rechaza PII; reglas
deterministas y versionadas dan una traza auditable (postura AI Act / DORA).

**P: ¿Qué harías a continuación?**
(1) Fotos reales de parking (cierra la brecha de dominio — la mayor palanca). (2) Eval de negocio
en golden set. (3) Baremo/precios de pieza reales. (4) Fusionar/arreglar `paint_chip`.

---

## 11. Resumen en un párrafo (si tienes 30 segundos)

> Un front-end de visión de dos modelos (un segmentador de **daños** YOLOv11 de 6 clases y un
> segmentador de **partes** de 23 clases) localiza y tipifica el daño del vehículo, y después una
> capa de reglas **determinista y auditable** lo convierte en una estimación de coste y una
> decisión de triaje verde/ámbar/rojo. La visión nunca decide dinero ni enrutamiento — solo
> propone candidatos — así que el sistema es explicable y seguro por construcción. El modelo es
> **experimental**: construido sobre datos públicos con etiquetas auto-generadas y precios
> placeholder; el camino a producción son fotos reales de dominio, el baremo oficial y una
> evaluación en golden set.
