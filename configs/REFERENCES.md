# Tablas de referencia económica — fuentes y mantenimiento (T2.1)

Este documento describe **de dónde sale cada tabla** que alimenta la estimación
de coste (`scripts/estimate_cost.py`, T2.2), **qué cobertura mínima garantiza**
y **cómo se actualiza**. Las tres tablas viven en `configs/` y se validan en
`tests/test_reference_tables.py`.

> ⚠️ **TODOS los valores actuales son PLACEHOLDER.**
> Versión `0.1.0-PLACEHOLDER`, `status: DRAFT`. Están basados en rangos públicos
> orientativos del sector y **NO deben usarse para liquidar siniestros reales**
> hasta sustituirlos por datos pactados con Mutua y firmar el cambio. Cada tabla
> lleva el marcador de placeholder en su cabecera; los tests fallan si se borra
> sin subir a una versión no-placeholder.

---

## 1. `baremo_horas.yaml` — horas de mano de obra por reparación

| Campo | Valor |
|---|---|
| **Qué contiene** | Horas de chapa y pintura, y la decisión `repair / repaint_only / replace`, por `(categoría_pieza, tipo_daño, extensión)`. |
| **Fuente (placeholder)** | Rangos públicos orientativos: observatorio **CETRAA**, **Centro Zaragoza**, baremos publicados por aseguradoras. |
| **Fuente (objetivo productivo)** | Baremo oficial pactado con Mutua / red de talleres propia. |
| **Lo consume** | T2.2: `coste = (chapa_h + pintura_h) · tarifa_provincia + precio_pieza (si replace) + materiales_pintura (15% de la mano de pintura)`. |

**Estructura**: `baremo[part_category][damage_type][extension] = {chapa_h, pintura_h, decision}`.

**Vocabularios** (validados contra el schema y entre sí):
- `part_category` ⊆ enum `damages[].part_category` de `schemas/inference_output_v1.json`
  (`plastic_panel, body_panel, light_assembly, mirror, glass, wheel, unknown`).
- `decision` ⊆ enum `damages[].repair_decision` del schema (`repair, repaint_only, replace`).
- `damage_type` ∈ `{scratch, dent, crack, broken, any}` · `extension` ∈ `{small, medium, large, any}`.

**Notas de mapeo / huecos conocidos**:
- El tipo `broken` del baremo mapea a la clase `broken_light` del modelo (el baremo
  es genérico por categoría de pieza, no por clase del detector).
- **`glass` no está en el baremo a propósito**: lunas/cristales están *fuera de
  alcance* del sistema (ver `implementation_plan.md`), van a flujo aparte.
- `unknown` existe como fallback conservador para piezas no identificadas.

**Cobertura mínima garantizada** (asegurada por test):
- Categorías reparables principales `plastic_panel` y `body_panel` cubren `scratch`, `dent` y `crack`.
- Toda categoría tiene al menos una entrada de daño.
- `paint_materials.pct_of_pintura_h_cost` definido; `additional_operations` con tiempos ≥ 0.

---

## 2. `precios_taller.yaml` — tarifas €/h por provincia

| Campo | Valor |
|---|---|
| **Qué contiene** | €/h de chapa y pintura por provincia + `default` nacional, modificadores (taller concertado, premium, urgencia) e IVA. |
| **Fuente (placeholder)** | Medias del observatorio **CETRAA / Centro Zaragoza**; diferencias inter-provinciales orientativas (hasta ~40%). |
| **Fuente (objetivo productivo)** | Tarifas pactadas con la red de talleres de Mutua. |
| **Lo consume** | T2.2: tarifa por hora según `metadata.provincia` (match case-insensitive; si no está → `default`). |

**Convenciones**:
- `currency: EUR`, `iva_included: false` → las tarifas son **sin IVA**; el estimador
  añade el **21%** (`vat.rate_pct`) al final.
- Pintura €/h ≥ chapa €/h en todas las filas (materiales/cabina), validado por test.

**Cobertura mínima garantizada** (asegurada por test):
- Existe `default` (fallback) con `chapa_eur_h` y `pintura_eur_h` > 0.
- Provincias mínimas presentes: **Madrid, Barcelona, Valencia, Sevilla, Zaragoza, Málaga, Bizkaia**.
- `vat.rate_pct == 21.0` (tipo general vigente en España).

---

## 3. `piezas.yaml` — catálogo de precios de piezas

| Campo | Valor |
|---|---|
| **Qué contiene** | Precio OEM y aftermarket por `marca → modelo → familia_pieza (+ rango de años, tech)`, política OEM/aftermarket y `fallback_prices`. |
| **Fuente (placeholder)** | Rangos públicos de recambios online y catálogos OEM 2024-2025. |
| **Fuente (objetivo productivo)** | Catálogo propio o API de proveedor: **Audatex**, **GT Motive**, **DAT Iberia**. |
| **Lo consume** | T2.2: precio de pieza cuando `decision == replace`. Si la combinación no está catalogada → alerta `parts_lookup_missing`, precio desde `fallback_prices`, y el caso va a **ámbar** (no puede ir a verde sin precio fiable). |

**Estructura**: `catalog[marca][modelo][familia_pieza] = {year_range:[a,b], oem_eur, aftermarket_eur(null si no hay), tech, paint_required?}`.

**Marcas cubiertas** (top 6 del parque español, DGT 2024-2025): **SEAT, Renault,
Peugeot, Volkswagen, Toyota, Ford** — exigidas por el plan.

**Cobertura mínima garantizada** (asegurada por test):
- Las 6 marcas anteriores presentes.
- ≥ 20 entradas de pieza catalogadas en total (el plan pide "top 20").
- `fallback_prices` cubre familias núcleo: paragolpes (delantero/trasero), una puerta, un faro, un retrovisor y rueda.
- `aftermarket_eur` es `null` o > 0; `oem_eur` > 0; `year_range` coherente (`a ≤ b`).

---

## Proceso de actualización (gobernanza)

1. **Sustituir valores** por los datos reales (baremo pactado, tarifas de red,
   catálogo/API de piezas).
2. **Subir versión** (`version`) y `effective_date` de la tabla modificada, y
   retirar el sufijo `-PLACEHOLDER` / el `status: DRAFT` cuando los datos sean
   reales (los tests exigen que, mientras sean placeholder, el marcador esté).
3. **Registrar en `data_lineage.yaml`** (T5.3): origen, fecha, hash y responsable
   del dato (regla 12 de trazabilidad).
4. **Reflejar en la model card** la versión de tablas usada en cada evaluación
   de negocio (el audit log ya guarda `rules_versions`).
5. **Revisar tras 100 siniestros reales**: contrastar el coste estimado contra el
   importe finalmente pagado (MAE €, Sprint 3) y recalibrar.

> Estas tablas afectan a decisiones económicas y de carril (verde/ámbar/rojo):
> cualquier cambio es un cambio de *compliance* y debe quedar versionado y firmado
> antes de uso productivo.
