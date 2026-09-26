"""
run_experiment.py
=================
Script principal del experimento Hispano-Bias v2.0.

Orquesta el pipeline completo de inferencia real:
  1. Inicialización del cliente Groq (LlamaInference) y motor de embeddings
  2. Iteración sobre los 30 pares del corpus con progreso visual (tqdm)
  3. 4 llamadas a la API por par: control peninsular, control latam,
     experimental peninsular, experimental latam
  4. Cálculo de deltas de sesgo y reducción porcentual
  5. Guardado de resultados en CSV y generación de gráficos

IMPORTANTE: Este script realiza llamadas REALES a la API de Groq.
No hay simulaciones ni datos precalculados. El experimento completo
implica aproximadamente 120 llamadas a la API (~15–30 minutos).

Uso:
    python run_experiment.py

Prerrequisito:
    Configurar GROQ_API_KEY en el archivo .env (ver .env.example)

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
Curso: Seminario de Investigación 2 (2026-1)
"""

import os
import sys
from datetime import datetime

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

# Cargar variables de entorno antes de importar módulos que las necesiten
load_dotenv()

# Importar módulos del proyecto
from corpus.corpus_v2 import CORPUS_V2
from pipeline.inference import LlamaInference
from pipeline.embeddings import EmbeddingEngine
from pipeline.metrics import build_results_dataframe, print_summary_table
from visualization.plots import plot_delta_by_domain, plot_reduction_heatmap


def imprimir_banner():
    """Imprime el banner de inicio del experimento con metadatos clave."""
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 64)
    print("   EXPERIMENTO HISPANO-BIAS v2.0")
    print("   Evaluación del Sesgo Lingüístico Digital en LLMs")
    print("   Rodrigo Alonso Gálvez Arrascue — ULima 2026")
    print("=" * 64)
    print("  Fecha y hora de inicio: " + ahora)
    print("  Pares en el corpus:     " + str(len(CORPUS_V2)))
    print("  Llamadas API totales:   ~" + str(len(CORPUS_V2) * 4))
    print("  Modelo LLM:             llama-3.1-8b-instant (Groq API)")
    print("  Modelo embeddings:      paraphrase-multilingual-MiniLM-L12-v2")
    print("=" * 64 + "\n")


def guardar_resultados_parciales(results, ruta):
    """
    Guarda los resultados acumulados hasta el momento de la interrupción.

    Parámetros
    ----------
    results : list[dict]
        Lista de resultados parciales.
    ruta : str
        Ruta del CSV de salida.
    """
    if not results:
        print("\n[Ctrl+C] No hay resultados parciales para guardar.")
        return

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df_parcial = build_results_dataframe(results)
    df_parcial.to_csv(ruta, index=False, encoding="utf-8")
    print("\n[Ctrl+C] Experimento interrumpido.")
    print("[Ctrl+C] Se guardaron " + str(len(results)) + " resultados parciales en:")
    print("[Ctrl+C] " + ruta)


if __name__ == "__main__":

    # ── 1. Banner de inicio ───────────────────────────────────────────────────
    imprimir_banner()

    # ── 2. Inicializar componentes del pipeline ───────────────────────────────
    print("[Paso 1/5] Inicializando LlamaInference (cliente Groq)...")
    llm = LlamaInference()

    print("\n[Paso 2/5] Inicializando EmbeddingEngine (sentence-transformers)...")
    embeddings = EmbeddingEngine()

    print("\n[Paso 3/5] Iniciando iteración sobre el corpus...\n")

    # Lista donde se acumulan los resultados de cada par procesado exitosamente
    results = []

    # Ruta del CSV de resultados parciales (para recuperación ante interrupción)
    RUTA_PARCIAL = "results/data/resultados_parciales.csv"

    try:
        # ── 3. Iterar sobre el corpus con barra de progreso ───────────────────
        for par in tqdm(CORPUS_V2, desc="Procesando corpus", unit="par"):

            item_id      = par["id"]
            texto_pen    = par["peninsular"]
            texto_lat    = par["latam"]
            dominio      = par["dominio"]
            categoria    = par["categoria"]

            tqdm.write("\n[ID " + str(item_id) + "] " + dominio + " — " + categoria)

            # ── Condición CONTROL ─────────────────────────────────────────────
            # Dos llamadas separadas: una por variedad dialectal
            resp_ctrl_pen = llm.generate_control(texto_pen, item_id=item_id)
            resp_ctrl_lat = llm.generate_control(texto_lat, item_id=item_id)

            # ── Condición EXPERIMENTAL (con PET) ─────────────────────────────
            resp_exp_pen  = llm.generate_experimental(texto_pen, item_id=item_id)
            resp_exp_lat  = llm.generate_experimental(texto_lat, item_id=item_id)

            # ── Verificar que ninguna llamada falló ───────────────────────────
            if any(r is None for r in [resp_ctrl_pen, resp_ctrl_lat, resp_exp_pen, resp_exp_lat]):
                tqdm.write(
                    "[ADVERTENCIA] ID " + str(item_id)
                    + ": una o más llamadas a la API fallaron. Se omite este par."
                )
                continue

            # ── Calcular deltas de sesgo con embeddings ────────────────────────
            delta_base     = embeddings.compute_delta(resp_ctrl_pen, resp_ctrl_lat)
            delta_mitigado = embeddings.compute_delta(resp_exp_pen,  resp_exp_lat)

            tqdm.write(
                "  Delta base: " + str(round(delta_base, 4))
                + "  |  Delta mitigado: " + str(round(delta_mitigado, 4))
            )

            # ── Añadir resultado completo a la lista ──────────────────────────
            results.append({
                "id":                        item_id,
                "item_peninsular":           texto_pen,
                "item_latam":                texto_lat,
                "dominio":                   dominio,
                "categoria":                 categoria,
                "respuesta_control_pen":     resp_ctrl_pen,
                "respuesta_control_lat":     resp_ctrl_lat,
                "respuesta_experimental_pen": resp_exp_pen,
                "respuesta_experimental_lat": resp_exp_lat,
                "delta_base":                delta_base,
                "delta_mitigado":            delta_mitigado,
            })

    except KeyboardInterrupt:
        # ── Guardar resultados parciales ante interrupción manual ─────────────
        guardar_resultados_parciales(results, RUTA_PARCIAL)
        sys.exit(0)

    # ── 4. Construir DataFrame y mostrar tabla resumen ────────────────────────
    if not results:
        print("\n[ERROR] No se pudo procesar ningún ítem del corpus.")
        print("Verifica tu conexión a Internet y la validez de la GROQ_API_KEY.")
        sys.exit(1)

    print("\n[Paso 4/5] Construyendo DataFrame de resultados...")
    df = build_results_dataframe(results)

    print("\n[Paso 4/5] Tabla de resultados:")
    print_summary_table(df)

    # ── 5. Guardar resultados en disco ────────────────────────────────────────
    print("[Paso 5/5] Guardando resultados...")
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # CSV principal (sin columnas de texto de respuesta)
    columnas_principales = [
        "id", "item_peninsular", "item_latam", "dominio", "categoria",
        "delta_base", "delta_mitigado", "reduccion_pct",
    ]
    ruta_csv = "results/data/resultados_v2.csv"
    df[columnas_principales].to_csv(ruta_csv, index=False, encoding="utf-8")
    print("  -> " + ruta_csv)

    # CSV completo (con todas las respuestas del LLM)
    ruta_csv_completo = "results/data/resultados_v2_completo.csv"
    df.to_csv(ruta_csv_completo, index=False, encoding="utf-8")
    print("  -> " + ruta_csv_completo)

    # ── 6. Generar gráficos ───────────────────────────────────────────────────
    print("\n[Paso 5/5] Generando gráficos...")
    plot_delta_by_domain(df, "results/figures/delta_por_dominio.png")
    plot_reduction_heatmap(df, "results/figures/heatmap_reduccion.png")

    # ── 7. Mensaje final ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("   EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print("=" * 64)
    print("Archivos generados:")
    print("  " + ruta_csv)
    print("  " + ruta_csv_completo)
    print("  results/figures/delta_por_dominio.png")
    print("  results/figures/heatmap_reduccion.png")
    print("\nPróximo paso:")
    print("  python run_iem_analysis.py")
    print("=" * 64 + "\n")
