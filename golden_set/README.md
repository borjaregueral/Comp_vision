# Golden Set — siniestros reales de parking (T3.1)

El golden set es el conjunto de **siniestros cerrados reales** que se usa para medir
las métricas de negocio (T3.3: MAE €, kappa, FN estructural…) y para evaluar la
versión `baseline_v1.0` del modelo (T3.5). Sin él no sabemos si el sistema funciona
en producción.

> 🔒 **PII — lectura obligatoria.**
> Este directorio (`golden_set/`) está **gitignored** salvo este README y `.gitkeep`.
> **NUNCA** se versionan en el repo: imágenes de asegurados, matrículas, nombres,
> DNI, direcciones, partes reales. Los datos viven en **almacenamiento cifrado fuera
> del repo**. El esquema de ground truth (`schemas/ground_truth_v1.json`) usa
> `additionalProperties:false` en todos los niveles para que cualquier campo PII
> haga **fallar la validación** (defensa estructural, no solo política).

> ⏳ **Estado: BLOQUEADO en coordinación con Mutua.**
> Esta tarea (T3.1) entrega la **especificación**: criterios, esquema y proceso de
> anonimización. La **obtención de los datos** es coordinación con Mutua/Legal.
> Con la spec lista, T3.2 (loader) se construye y testea con un golden set sintético.

---

## 1. Criterios de selección

| Criterio | Valor |
|---|---|
| Tipo | Siniestros de **aparcamiento** (daños propios, sin terceros heridos) |
| Estado | **Cerrados** (importe final conocido) |
| Importe | Mayoritariamente **< 1.500 €**; incluir un grupo **> 1.500 €** como *control* |
| Antigüedad | **Último año** (`fecha_cierre`) |
| Tamaño | **500–1.000** siniestros |
| Fuente | Extracto de cartera de Mutua (a coordinar) |

**Estratificación obligatoria** por tramo de importe real:
- `< 500 €`, `500–1.500 €`, `> 1.500 €` (control: estos **deberían** ir a carril rojo).

Buscar además variedad de **marca, color y provincia** para poder hacer bias testing
(T5.2) y que las métricas no estén sesgadas a un segmento.

## 2. Estructura en disco (todo gitignored salvo este README)

```
golden_set/
├── README.md                      (versionado)
├── ground_truth/<claim_id>.json   (1 por siniestro, valida contra ground_truth_v1.json)
└── images/<claim_id>/*.jpg        (fotos ya anonimizadas)
```

## 3. Esquema de ground truth

Definido en **`schemas/ground_truth_v1.json`** (canónico). Campos principales:

| Campo | Uso |
|---|---|
| `claim_id` | ID pseudónimo (no PII) |
| `importe_final_pagado` (+`moneda`,`iva_incluido`) | **Target del MAE €** |
| `severidad_oficial` (`leve/moderado/severo`) | Cohen's weighted kappa vs sistema |
| `es_estructural` | Tasa de **FN estructural** (≤2% innegociable) |
| `decision_final` | Acuerdo de carril (ver mapeo abajo) |
| `piezas[]`, `horas_reales` | Desglose real (validar baremo/piezas en Sprint 4) |
| `vehiculo{marca,modelo,anio,color,color_grupo,provincia,valor_estimado}` | Estratificación + bias |
| `fecha_cierre` | Filtro de "último año" |
| `perito_id` | Análisis inter-perito (pseudónimo) |

**Mapeo `decision_final` ↔ carril del sistema** (para medir acuerdo):
- `resuelto_sin_peritaje` ↔ **verde**
- `revisado_por_tramitador` ↔ **ámbar**
- `peritaje_presencial` ↔ **rojo**
- `rechazado` ↔ fuera de gestión automática

## 4. Proceso de anonimización (antes de que nada entre al repo/almacén)

1. **EXIF/GPS**: eliminar con `scripts/quality_gate.extract_and_strip_exif` (RGPD).
2. **Matrícula**: difuminar/recortar en todas las fotos (revisión manual o detector).
3. **Identificadores**: sustituir el nº de siniestro real por un `claim_id` pseudónimo;
   mantener el mapeo real→pseudónimo **solo** en el almacén cifrado, fuera del repo.
4. **Sin PII en el JSON**: el esquema (`additionalProperties:false`) rechaza campos no
   declarados; revisar `notas` para que no contenga datos personales.
5. **Responsable y base legal**: registrar quién anonimiza, cuándo, y la base RGPD del
   tratamiento.

## 5. Gobernanza y trazabilidad

- Registrar el golden set en **`data_lineage.yaml`** (T5.3): origen, fecha de ingesta,
  hash del extracto, responsable.
- La versión del golden set usada en cada evaluación se cita en la **model card** (T3.5).
- Retención y borrado conforme a la política RGPD acordada con Legal.

## 6. Qué desbloquea

- **T3.2** (`load_golden_set.py`): carga + valida contra `ground_truth_v1.json` +
  estratifica por tramo. Construible/testeable **ya** con datos sintéticos.
- **T3.5** (evaluación de `baseline_v1.0`): requiere el golden set **real** + `best.pt`.
