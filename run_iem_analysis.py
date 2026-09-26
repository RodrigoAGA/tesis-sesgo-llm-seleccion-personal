"""
run_iem_analysis.py
===================
Script secundario para el cálculo del Índice de Estabilidad de Mitigación (IEM).

Este script carga los resultados reales generados por run_experiment.py y calcula:
  - Promedios de reducción del sesgo por dominio (D1, D2, D3)
  - IEM = σ_inter / |μ_global| (índice de consistencia inter-dominio, corregido)
  - IEM robusto (excluyendo ítems con |delta_base| < 0.01)
  - Test de Kruskal-Wallis (H0: distribuciones equivalentes entre dominios)

PREREQUISITO: Ejecutar primero run_experiment.py para generar los resultados.

Uso:
    python run_iem_analysis.py

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
Curso: Seminario de Investigación 2 (2026-1)
"""

import os
import sys
import json

import pandas as pd

from analysis.iem import compute_iem
from visualization.plots import plot_iem_consistency

# ── Rutas de archivos ─────────────────────────────────────────────────────────
RUTA_CSV      = "results/data/resultados_v2.csv"
RUTA_JSON_IEM = "results/data/iem_summary.json"
RUTA_PLOT_IEM = "results/figures/iem_consistencia.png"


def verificar_prerequisitos(df):
    """
    Verifica que el DataFrame cargado tiene los dominios y columnas necesarios.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame cargado desde el CSV de resultados.

    Retorna
    -------
    bool
        True si todos los dominios están presentes, False si alguno falta
        (con advertencia en consola, pero sin detener la ejecución).
    """
    columnas_requeridas = ["id", "dominio", "reduccion_pct", "delta_base"]
    for col in columnas_requeridas:
        if col not in df.columns:
            print("[ERROR] El CSV no contiene la columna requerida: " + col)
            print("El archivo puede estar corrupto o ser de una versión anterior.")
            sys.exit(1)

    dominios_presentes  = set(df["dominio"].unique())
    dominios_requeridos = {"D1", "D2", "D3"}
    dominios_faltantes  = dominios_requeridos - dominios_presentes

    if dominios_faltantes:
        print(
            "[ADVERTENCIA] El CSV no contiene ítems de los dominios: "
            + ", ".join(sorted(dominios_faltantes))
        )
        print(
            "El cálculo del IEM puede ser incompleto si no están presentes los tres dominios."
        )
        return False

    return True


def contar_por_dominio(df):
    """
    Cuenta el número de ítems procesados por cada dominio.

    Parámetros
    ----------
    df : pandas.DataFrame

    Retorna
    -------
    dict
        Diccionario con conteos: {"D1": N, "D2": N, "D3": N}
    """
    conteos = {}
    for dom in ["D1", "D2", "D3"]:
        conteos[dom] = len(df[df["dominio"] == dom])
    return conteos


def imprimir_resumen_ejecutivo(df, iem_results, conteos):
    """
    Imprime el resumen ejecutivo del IEM en el formato estándar de la tesis,
    incluyendo el bloque de análisis de robustez.

    Parámetros
    ----------
    df : pandas.DataFrame
    iem_results : dict
        Resultado de compute_iem().
    conteos : dict
        Número de ítems por dominio.
    """
    linea       = "=" * 60
    linea_corta = "-" * 52

    # Extraer valores del diccionario IEM
    mu_D1       = iem_results["mu_D1"]
    mu_D2       = iem_results["mu_D2"]
    mu_D3       = iem_results["mu_D3"]
    mu_global   = iem_results["mu_global"]
    sigma_inter = iem_results["sigma_inter"]
    iem         = iem_results["IEM"]
    iem_robusto = iem_results.get("iem_robusto")
    n_excluidos = iem_results.get("n_excluidos_robustez", 0)
    kruskal_H   = iem_results["kruskal_H"]
    kruskal_p   = iem_results["kruskal_p"]
    interp_iem  = iem_results["interpretacion_IEM"]
    interp_krus = iem_results["interpretacion_kruskal"]
    direccion   = iem_results.get("direccion_mitigacion", "desconocida")

    total = sum(conteos.values())

    # ── Encabezado ────────────────────────────────────────────────────────────
    print("\n" + linea)
    print("   ÍNDICE DE ESTABILIDAD DE MITIGACIÓN (IEM)")
    print("   Corpus Hispano-Bias v2.0 — Rodrigo Gálvez — ULima 2026")
    print(linea)

    # ── Dirección del efecto global ───────────────────────────────────────────
    print()
    print("DIRECCIÓN DEL EFECTO GLOBAL DEL PET:")
    signo = "+" if mu_global >= 0 else ""
    print("  " + signo + str(round(mu_global, 2)) + "% — " + direccion + " del sesgo")

    # ── Ítems por dominio ─────────────────────────────────────────────────────
    print("\nÍtems procesados por dominio:")
    print("  D1 — Léxico Coloquial:          " + str(conteos.get("D1", 0)) + " ítems")
    print("  D2 — Terminología Profesional:  " + str(conteos.get("D2", 0)) + " ítems")
    print("  D3 — Morfosintaxis Formal:      " + str(conteos.get("D3", 0)) + " ítems")
    print("  Total:                          " + str(total) + " ítems")

    # ── Reducción por dominio ─────────────────────────────────────────────────
    print("\nReducción promedio del sesgo por dominio:")
    print("  D1 — Léxico Coloquial:          " + str(round(mu_D1, 2)) + "%")
    print("  D2 — Terminología Profesional:  " + str(round(mu_D2, 2)) + "%")
    print("  D3 — Morfosintaxis Formal:      " + str(round(mu_D3, 2)) + "%")
    print("  Global:                         " + str(round(mu_global, 2)) + "%")

    # ── IEM ───────────────────────────────────────────────────────────────────
    print("\nDesviación estándar inter-dominio (σ): " + str(round(sigma_inter, 4)))
    print(
        "IEM = σ / |μ_global| = " + str(round(iem, 4))
        + "   [" + str(round(sigma_inter, 4))
        + " / " + str(round(abs(mu_global), 4)) + "]"
    )

    print("\nInterpretación IEM:")
    print("  → " + interp_iem)

    # ── Kruskal-Wallis ────────────────────────────────────────────────────────
    print(
        "\nTest de Kruskal-Wallis (H0: distribuciones equivalentes entre dominios):"
    )
    print(
        "  H = " + str(round(kruskal_H, 4))
        + "  |  p-value = " + str(round(kruskal_p, 4))
    )
    print("  → " + interp_krus)

    print("\n" + linea + "\n")

    # ── Bloque de robustez ────────────────────────────────────────────────────
    print(linea_corta)
    print("   ANÁLISIS DE ROBUSTEZ")
    print(linea_corta)
    print("Ítems excluidos (|delta_base| < 0.01): " + str(n_excluidos))
    print("IEM completo  (N=" + str(total) + "):    " + str(round(iem, 4)))

    n_robusto = total - n_excluidos
    if iem_robusto is None:
        print("IEM robusto   (N=" + str(n_robusto) + "):    N/A (datos insuficientes por dominio)")
        print("→ No es posible calcular el IEM robusto con los datos disponibles.")
    else:
        print("IEM robusto   (N=" + str(n_robusto) + "):    " + str(round(iem_robusto, 4)))
        diferencia = abs(iem - iem_robusto)
        if diferencia < 0.1:
            print(
                "→ Resultado robusto: el IEM no cambia significativamente al excluir "
                "ítems con delta_base cercano a cero"
            )
        else:
            print(
                "→ Nota metodológica: el IEM varía al excluir ítems con delta_base < 0.01 "
                "— reportar ambos valores en el documento"
            )

    print(linea_corta + "\n")


if __name__ == "__main__":

    # ── 1. Verificar que los resultados de la Fase 1 existen ─────────────────
    if not os.path.exists(RUTA_CSV):
        print("\n[ERROR] No se encontró el archivo de resultados:")
        print("  " + RUTA_CSV)
        print("\nDebes ejecutar primero el script principal:")
        print("  python run_experiment.py")
        sys.exit(1)

    # ── 2. Cargar resultados del experimento ──────────────────────────────────
    print("[run_iem_analysis] Cargando resultados desde: " + RUTA_CSV)
    df = pd.read_csv(RUTA_CSV, encoding="utf-8")
    print("[run_iem_analysis] Ítems cargados: " + str(len(df)))

    # ── 3. Verificar integridad del DataFrame ─────────────────────────────────
    verificar_prerequisitos(df)
    conteos = contar_por_dominio(df)

    # ── 4. Calcular el IEM (completo + robusto) ───────────────────────────────
    print("[run_iem_analysis] Calculando IEM y análisis de robustez...")
    iem_results = compute_iem(df)

    # ── 5. Imprimir resumen ejecutivo + bloque de robustez ────────────────────
    imprimir_resumen_ejecutivo(df, iem_results, conteos)

    # ── 6. Guardar resumen JSON actualizado ───────────────────────────────────
    # Incluye los campos extendidos: iem_corregido, iem_robusto,
    # n_excluidos_robustez y direccion_mitigacion
    json_payload = dict(iem_results)
    json_payload["iem_corregido"] = iem_results["IEM"]   # alias explícito

    os.makedirs("results/data", exist_ok=True)
    with open(RUTA_JSON_IEM, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print("[run_iem_analysis] Resumen JSON guardado: " + RUTA_JSON_IEM)

    # ── 7. Generar gráfico de consistencia IEM ────────────────────────────────
    os.makedirs("results/figures", exist_ok=True)
    plot_iem_consistency(iem_results, RUTA_PLOT_IEM)

    # ── 8. Mensaje de archivos generados ─────────────────────────────────────
    print("\nArchivos generados en esta fase:")
    print("  " + RUTA_JSON_IEM)
    print("  " + RUTA_PLOT_IEM)
    print("\nAnálisis IEM completado.\n")
