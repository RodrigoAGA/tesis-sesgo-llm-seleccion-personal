# Evaluación de Sesgo en LLMs para Selección de Personal — Tesis

**Repositorio de Tesis — Ingeniería de Sistemas — Universidad de Lima**

> **Investigador:** Rodrigo Alonso Gálvez Arrascue  
> **Institución:** Universidad de Lima — Carrera de Ingeniería de Sistemas  
> **Curso:** Seminario de Investigación 2 (2026-2)  
> **Última actualización:** Septiembre 2026

---

## Estado actual del proyecto (Septiembre 2026)

La tesis propone un **protocolo e instrumento automatizado para evaluar el
sesgo de modelos de lenguaje (LLMs) aplicados a procesos de selección de
personal** (screening de CVs, ranking de candidatos, entrevistas). El
instrumento se conecta por API a distintos proveedores de LLM, les aplica
una batería de pruebas contrafácticas (candidatos equivalentes que solo
difieren en un atributo protegido) y clasifica a cada modelo como **apto /
no apto** según los umbrales del protocolo propuesto.

El alcance es **solo de evaluación**: un modelo que no pasa el protocolo se
descarta y se evalúa otro; no se intenta corregirlo. El entregable es una
metodología reutilizable, validada empíricamente, no un estándar adoptado
por la industria.

Motivación normativa: en Perú, el Reglamento de la Ley 31814 (DS
115-2025-PCM) clasifica como de **riesgo alto** a los sistemas de IA que
intervienen en la selección, evaluación, contratación y cese de
trabajadores o postulantes, y exige evaluar su impacto y auditar sus sesgos.

Piezas ya construidas:

- `pipeline/llm_connector.py` — conector multi-LLM con una interfaz común
  (`LLMProvider.generate()`). Groq y Gemini funcionando; OpenAI y Claude
  quedan como adaptadores pendientes de API key de pago.
- `demo_multi_llm.py` — prueba de concepto que manda el mismo prompt a
  varios proveedores y compara sus respuestas.
- `figuras/arquitectura-multi-llm-rrhh.drawio` (+ `.png`) — arquitectura
  del instrumento, de la batería de pruebas a la clasificación final.

En definición: el dataset de candidatos (debe incluir diversidad de edad,
género y lugar de nacimiento; *Bias in Bios* — De-Arteaga et al., 2019 — y
*JobFair* — Wang et al., 2024 — no cubren edad ni procedencia) y la batería
de pruebas específica de RRHH.

El resto de este documento describe el **enfoque anterior de la
investigación (sesgo lingüístico dialectal)**, que es el antecedente
directo del instrumento actual: reutiliza el mismo pipeline base de
inferencia, embeddings y métricas.

---

## Antecedente: sesgo lingüístico dialectal

El experimento evalúa si un LLM asocia distintos niveles de competencia
profesional a textos equivalentes en contenido pero escritos en distintas
variedades del español (peninsular vs. latinoamericano), y si un prompt de
mitigación (PET — Prompt de Estabilización Terminológica) reduce esa
diferencia de forma consistente entre dominios semánticos.

### Metodología

1. **Corpus Hispano-Bias v2.0**: 30 pares de estímulos con equivalencia semántica estricta, organizados en 3 dominios:
   - **D1 — Léxico Coloquial** (IDs 1–10)
   - **D2 — Terminología Profesional** (IDs 11–20)
   - **D3 — Morfosintaxis Formal** (IDs 21–30)

2. **Inferencia real** vía Groq API: cada estímulo se procesa en condición **control** (sin PET) y **experimental** (con PET).

3. **Similitud coseno** con un texto de referencia ("Alta competencia profesional, formalidad, éxito corporativo, prestigio académico.") usando el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2`.

4. **Índice de Estabilidad de Mitigación (IEM)**: métrica que cuantifica la consistencia de la reducción del sesgo entre dominios semánticos.

**Modelo evaluado:** `openai/gpt-oss-20b` vía Groq, con `reasoning_effort="low"`.
El diseño original usaba Llama-3-8B-Instruct, pero toda la familia Llama 3
fue retirada del catálogo de Groq (verificado en septiembre de 2026).

**Resultados de la corrida piloto** (`results/data/`): inferencia real, 120
llamadas, sin respuestas vacías. No se observa un efecto consistente del
PET: 15 de 30 ítems reducen la diferencia y 15 la aumentan (mediana de
|Δ| 0.0379 sin PET y 0.0293 con PET). La reducción porcentual es inestable
cuando la diferencia base es cercana a cero, por lo que se reportan
magnitudes absolutas.

Los notebooks de `notebooks/` son exploración inicial y contienen valores
simulados; no deben leerse como resultados.

---

## Requisitos Previos

- **Python 3.9+**
- API key gratuita de [Groq](https://console.groq.com) y, para el conector multi-LLM, de [Google AI Studio](https://aistudio.google.com/apikey) (Gemini).
- Conexión a Internet (inferencia vía API y descarga del modelo de embeddings en el primer uso).

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/RodrigoAGA/tesis-sesgo-llm-seleccion-personal.git
cd tesis-sesgo-llm-seleccion-personal

# 2. Crear y activar el entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar las API keys
cp .env.example .env
# Editar .env y reemplazar los valores de ejemplo por tus keys reales
```

---

## Cómo Ejecutar

### Conector multi-LLM

```bash
python demo_multi_llm.py
```

### Experimento dialectal (antecedente)

```bash
python run_experiment.py      # Inferencia + embeddings + métricas (≈ 15–30 min)
python run_iem_analysis.py    # IEM + Kruskal-Wallis + gráficos (< 1 min)
```

`run_experiment.py` hace 120 llamadas a la API (4 por par: control
peninsular, control latam, experimental peninsular, experimental latam). Si
se interrumpe con `Ctrl+C`, los resultados parciales se guardan en
`results/data/resultados_parciales.csv`.

| Archivo | Descripción |
|---|---|
| `results/data/resultados_v2.csv` | Deltas y reducción por ítem |
| `results/data/resultados_v2_completo.csv` | Resultados completos, con el texto de cada respuesta del LLM |
| `results/data/iem_summary.json` | Resumen del IEM con sus estadísticos |
| `results/figures/delta_por_dominio.png` | Delta base vs. delta mitigado por ítem y dominio |
| `results/figures/heatmap_reduccion.png` | Mapa de calor de reducción por ítem y dominio |
| `results/figures/iem_consistencia.png` | Reducción promedio por dominio |

---

## Estructura del Repositorio

```
tesis-sesgo-llm-seleccion-personal/
├── README.md
├── requirements.txt
├── .env.example                 # Plantilla de variables de entorno
├── demo_multi_llm.py            # Prueba del conector multi-LLM
├── run_experiment.py            # Experimento dialectal: inferencia + métricas
├── run_iem_analysis.py          # Experimento dialectal: IEM + pruebas estadísticas
│
├── pipeline/
│   ├── llm_connector.py         # Conector multi-LLM (Groq, Gemini; OpenAI/Claude pendientes)
│   ├── inference.py             # Cliente de inferencia vía Groq
│   ├── embeddings.py            # EmbeddingEngine (sentence-transformers)
│   └── metrics.py               # Cálculo de métricas
│
├── corpus/corpus_v2.py          # 30 pares de estímulos del experimento dialectal
├── analysis/iem.py              # compute_iem() con Kruskal-Wallis
├── visualization/plots.py       # Gráficos
├── notebooks/                   # Exploración inicial (valores simulados)
├── figuras/                     # Diagramas de arquitectura y figuras de la tesis
└── results/                     # Datos y gráficos de la corrida piloto
```

---

*Universidad de Lima — Carrera de Ingeniería de Sistemas — 2026*
