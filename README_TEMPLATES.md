# Plantillas de configuración para Sprint 1-2 — Comp_vision

Este paquete contiene las plantillas mínimas de configuración y schemas que el agente necesita para ejecutar Sprint 1 (orquestación) y Sprint 2 (estimación económica) del `IMPLEMENTATION_PLAN.md` **sin tener que inventarse los campos sobre la marcha**.

## Contenido

```
configs/
├── quality_gate.yaml          # Umbrales del filtro de calidad de imagen (T1.1)
├── baremo_horas.yaml          # Horas de chapa/pintura por pieza×daño (T2.1, T2.2)
├── precios_taller.yaml        # Tarifas €/h por provincia (T2.1, T2.2)
└── piezas.yaml                # Catálogo de precios OEM/aftermarket (T2.1, T2.2)

business_rules/
├── lane_rules.yaml            # Reglas deterministas de triaje verde/ámbar/rojo (T1.3)
└── severity_matrix.yaml       # Matriz pieza×daño→severidad económica (T2.3)

schemas/
└── inference_output_v1.json   # JSON Schema canónico de salida (T1.2)

docs/
└── MODEL_CARD_TEMPLATE.md     # Plantilla a copiar a model_cards/v{X.Y}.md (T3.5)
```

## Cómo integrar en el repo

Desde la raíz del repositorio `Comp_vision`:

```bash
# Si descargaste el paquete a un directorio temporal:
cp -r templates_pack/configs/*       configs/
cp -r templates_pack/business_rules  ./
cp -r templates_pack/schemas         ./
cp templates_pack/docs/MODEL_CARD_TEMPLATE.md  docs/

# Verifica que parsean
python3 -c "
import json, yaml
yaml.safe_load(open('configs/quality_gate.yaml'))
yaml.safe_load(open('configs/baremo_horas.yaml'))
yaml.safe_load(open('configs/precios_taller.yaml'))
yaml.safe_load(open('configs/piezas.yaml'))
yaml.safe_load(open('business_rules/lane_rules.yaml'))
yaml.safe_load(open('business_rules/severity_matrix.yaml'))
json.load(open('schemas/inference_output_v1.json'))
print('Todo parsea OK')
"
```

## Estado de las plantillas

| Fichero | Estado | Qué falta para producción |
|---|---|---|
| `quality_gate.yaml` | **Production-ready** | Calibrar umbrales tras 100 fotos reales |
| `lane_rules.yaml` | **Production-ready** | Validar reglas con Tramitación de Mutua |
| `severity_matrix.yaml` | **Production-ready** | Validar matriz con perito senior |
| `inference_output_v1.json` | **Production-ready** | Ninguno, congelar como contrato v1 |
| `baremo_horas.yaml` | ⚠️ **PLACEHOLDER** | Sustituir por baremo oficial (CETRAA / red propia) |
| `precios_taller.yaml` | ⚠️ **PLACEHOLDER** | Sustituir por tarifas pactadas con red de talleres |
| `piezas.yaml` | ⚠️ **PLACEHOLDER** | Sustituir por catálogo real (Audatex / GT Motive / propio) |
| `MODEL_CARD_TEMPLATE.md` | **Production-ready** | Rellenar al entrenar cada versión |

Los tres ficheros marcados como PLACEHOLDER tienen valores orientativos basados en rangos públicos del sector. **El sistema puede arrancar y testarse end-to-end con ellos**, pero ningún siniestro real puede liquidarse en producción sin sustituirlos por datos auténticos.

## Cómo lo usa el agente

Cuando el agente ejecute las tareas del plan, debe:

1. **T1.1 (Quality Gate)** → leer `configs/quality_gate.yaml` para los umbrales.
2. **T1.2 (Schema)** → validar TODA salida contra `schemas/inference_output_v1.json` antes de emitirla.
3. **T1.3 (Triaje)** → cargar `business_rules/lane_rules.yaml` y aplicar reglas en orden (rojo → verde → ámbar). El código NO contiene umbrales hardcoded.
4. **T2.1-T2.2 (Estimación)** → consultar las tres tablas (`baremo_horas`, `precios_taller`, `piezas`) para construir la estimación. Si una pieza no está catalogada, generar alerta `parts_lookup_missing` y derivar a ámbar.
5. **T2.3 (Severidad)** → usar `severity_matrix.yaml`. NO usar la lógica naïve de `predict.py` por % de área de imagen.
6. **T3.5 (Model card)** → copiar `MODEL_CARD_TEMPLATE.md` a `model_cards/v1.0.md` y rellenarlo tras evaluación contra golden set.

## Reglas de modificación

1. **Versionado obligatorio**. Cualquier cambio sustantivo en un fichero requiere bump de `version` y nuevo `effective_date`.
2. **Nunca renombrar IDs**. Los IDs de reglas (`VERDE-1`, `ROJO-2`, etc.) y de alertas son contractuales. Para cambiar significado, crear ID nuevo y marcar el viejo como `deprecated`.
3. **Schema is contract**. `inference_output_v1.json` es el contrato con sistemas downstream. Si necesita cambios incompatibles, crear `v2.json` y mantener ambos durante la transición.
4. **Sin secretos en config**. Estos ficheros se versionan en Git. API keys, credenciales y datos PII van en `.env` (gitignored).

## Para revisión por stakeholders

Antes de Sprint 2 cerrado, conviene reunir con:

- **Tramitación**: validar reglas en `lane_rules.yaml` (umbrales económicos, condiciones de derivación).
- **Peritaje senior**: validar `severity_matrix.yaml` y catálogo de partes.
- **Compras / Talleres**: sustituir placeholders en `precios_taller.yaml` y `piezas.yaml` por datos negociados.
- **Legal**: revisar `inference_output_v1.json` (campos auditables) y `lane_rules.yaml` (criterios de derivación, cumplimiento AI Act).

## Siguiente paso

Una vez integrados estos ficheros en el repo, el agente puede arrancar Sprint 1 ejecutando:

```bash
# Tarea T0.1: verificar estado del entrenamiento
# Tarea T0.2: crear estructura de directorios nueva
# Tarea T0.3: setup de testing (pytest)
# Tarea T1.1: implementar quality_gate.py usando configs/quality_gate.yaml
# ...
```

El plan completo está en `IMPLEMENTATION_PLAN.md`. Las reglas de comportamiento del agente, en `AGENT_PROMPT.md`.
