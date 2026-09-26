"""
analysis/iem.py
===============
Cálculo del Índice de Estabilidad de Mitigación (IEM).

El IEM mide qué tan consistente es la reducción del sesgo producida por el PET
a lo largo de los tres dominios semánticos del corpus. Un IEM bajo indica que
la mitigación es homogénea entre dominios (evidencia de mitigación genuina);
un IEM alto indica heterogeneidad inter-dominio (posible supresión superficial).

Fórmula corregida:
    IEM = σ_inter / |μ_global|

donde:
    μ_Di      = promedio de reducción_pct en el dominio Di
    μ_global  = promedio global de reduccion_pct sobre todos los ítems
    σ_inter   = desviación estándar de [μ_D1, μ_D2, μ_D3] (ddof=1)

Se usa |μ_global| en el denominador para garantizar IEM ≥ 0 independientemente
de la dirección del efecto (reducción o amplificación del sesgo).
La dirección se reporta por separado en el campo 'direccion_mitigacion'.

Se complementa con el test de Kruskal-Wallis para verificar si las diferencias
inter-dominio son estadísticamente significativas.

Se incluye además el IEM robusto, calculado excluyendo ítems con
|delta_base| < 0.01 (ítems sin sesgo base apreciable que distorsionan
los porcentajes de reducción por efecto del denominador pequeño).

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import numpy as np
from scipy import stats


def _calcular_iem_sobre_df(df_filtrado):
    """
    Función auxiliar interna: calcula IEM, σ y μ sobre un DataFrame ya filtrado.
    Retorna None si algún dominio queda sin al menos 3 ítems.

    Parámetros
    ----------
    df_filtrado : pandas.DataFrame
        Subconjunto del DataFrame con columnas 'dominio' y 'reduccion_pct'.

    Retorna
    -------
    dict | None
        Diccionario con mu_D1, mu_D2, mu_D3, mu_global, sigma_inter, IEM,
        o None si no hay suficientes datos por dominio.
    """
    d1 = df_filtrado[df_filtrado["dominio"] == "D1"]["reduccion_pct"].tolist()
    d2 = df_filtrado[df_filtrado["dominio"] == "D2"]["reduccion_pct"].tolist()
    d3 = df_filtrado[df_filtrado["dominio"] == "D3"]["reduccion_pct"].tolist()

    # Verificar mínimo de 3 ítems por dominio para que la estadística sea válida
    if any(len(lst) < 3 for lst in [d1, d2, d3]):
        return None

    mu1 = float(np.mean(d1))
    mu2 = float(np.mean(d2))
    mu3 = float(np.mean(d3))
    mu_g = float(df_filtrado["reduccion_pct"].mean())
    sigma = float(np.std([mu1, mu2, mu3], ddof=1))

    if mu_g == 0.0:
        iem_val = 0.0
    else:
        iem_val = sigma / abs(mu_g)

    return {
        "mu_D1": mu1, "mu_D2": mu2, "mu_D3": mu3,
        "mu_global": mu_g, "sigma_inter": sigma, "IEM": iem_val,
    }


def _interpretar_iem(iem_val, direccion):
    """
    Genera el texto de interpretación del IEM incorporando la dirección del efecto.

    Parámetros
    ----------
    iem_val : float
        Valor del IEM corregido (≥ 0).
    direccion : str
        'amplificación' o 'reducción'.

    Retorna
    -------
    str
        Texto de interpretación cualitativa.
    """
    sufijo_dir = (
        " — el efecto global fue de " + direccion + " del sesgo"
        if direccion == "amplificación"
        else ""
    )

    if iem_val < 0.05:
        base = "Mitigación homogénea — evidencia de mitigación genuina"
    elif iem_val <= 0.10:
        base = "Heterogeneidad moderada — se requiere análisis adicional"
    else:
        base = (
            "Heterogeneidad significativa — evidencia de supresión superficial "
            "dominio-dependiente"
        )

    return base + sufijo_dir


def compute_iem(df):
    """
    Calcula el Índice de Estabilidad de Mitigación (IEM) a partir del DataFrame
    de resultados del experimento.

    Incluye el IEM completo (sobre todos los ítems) y el IEM robusto (excluyendo
    ítems donde |delta_base| < 0.01, que tienen sesgo base casi nulo y generan
    porcentajes de reducción extremos por efecto del denominador pequeño).

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame generado por build_results_dataframe() con columnas
        'dominio', 'reduccion_pct' y 'delta_base'.

    Retorna
    -------
    dict con las siguientes claves:
        mu_D1                : float — Reducción promedio en D1 (%)
        mu_D2                : float — Reducción promedio en D2 (%)
        mu_D3                : float — Reducción promedio en D3 (%)
        mu_global            : float — Reducción promedio global (%)
        sigma_inter          : float — Desviación estándar inter-dominio (ddof=1)
        IEM                  : float — IEM corregido = σ / |μ_global| (≥ 0)
        iem_robusto          : float | None — IEM sobre ítems con |Δbase| ≥ 0.01
        n_excluidos_robustez : int   — Ítems excluidos por el filtro de robustez
        direccion_mitigacion : str   — 'reducción' o 'amplificación'
        kruskal_H            : float — Estadístico H de Kruskal-Wallis
        kruskal_p            : float — p-valor del test de Kruskal-Wallis
        interpretacion_IEM   : str   — Interpretación cualitativa del IEM
        interpretacion_kruskal : str — Interpretación del test estadístico

    Lanza
    -----
    ValueError
        Si alguno de los dominios D1, D2 o D3 no tiene ítems en el DataFrame.
    """
    # ── Separar ítems por dominio para el IEM completo ────────────────────────
    d1_items = df[df["dominio"] == "D1"]["reduccion_pct"].tolist()
    d2_items = df[df["dominio"] == "D2"]["reduccion_pct"].tolist()
    d3_items = df[df["dominio"] == "D3"]["reduccion_pct"].tolist()

    # Validar que los tres dominios tienen datos
    dominios_faltantes = []
    if len(d1_items) == 0:
        dominios_faltantes.append("D1")
    if len(d2_items) == 0:
        dominios_faltantes.append("D2")
    if len(d3_items) == 0:
        dominios_faltantes.append("D3")

    if dominios_faltantes:
        raise ValueError(
            "El DataFrame no contiene ítems de los siguientes dominios: "
            + ", ".join(dominios_faltantes)
            + ". Asegúrate de que run_experiment.py haya procesado ítems de los tres dominios."
        )

    # ── Calcular estadísticos principales ────────────────────────────────────
    mu_D1 = float(np.mean(d1_items))
    mu_D2 = float(np.mean(d2_items))
    mu_D3 = float(np.mean(d3_items))

    mu_global = float(df["reduccion_pct"].mean())

    # Desviación estándar inter-dominio (ddof=1, estimador insesgado)
    sigma_inter = float(np.std([mu_D1, mu_D2, mu_D3], ddof=1))

    # ── IEM corregido = σ_inter / |μ_global| ─────────────────────────────────
    # Se usa el valor absoluto del denominador para garantizar IEM ≥ 0.
    # La dirección del efecto se captura en 'direccion_mitigacion'.
    if mu_global == 0.0:
        iem = 0.0
    else:
        iem = sigma_inter / abs(mu_global)

    # ── Dirección del efecto global del PET ───────────────────────────────────
    if mu_global >= 0:
        direccion_mitigacion = "reducción"
    else:
        direccion_mitigacion = "amplificación"

    # ── IEM robusto: excluir ítems con |delta_base| < 0.01 ───────────────────
    # Estos ítems tienen sesgo base casi nulo, y al calcular el % de reducción
    # se dividen valores pequeños generando outliers extremos (ej: -958%).
    UMBRAL_ROBUSTEZ = 0.01
    df_robusto = df[df["delta_base"].abs() >= UMBRAL_ROBUSTEZ].copy()
    n_excluidos_robustez = len(df) - len(df_robusto)

    resultado_robusto = _calcular_iem_sobre_df(df_robusto)
    if resultado_robusto is None:
        iem_robusto = None
        print(
            "[IEM] ADVERTENCIA: Tras excluir " + str(n_excluidos_robustez)
            + " ítems con |delta_base| < " + str(UMBRAL_ROBUSTEZ)
            + ", algún dominio tiene menos de 3 ítems. iem_robusto = None."
        )
    else:
        iem_robusto = resultado_robusto["IEM"]

    # ── Test de Kruskal-Wallis ────────────────────────────────────────────────
    # H0: las distribuciones de reducción son equivalentes entre dominios
    kruskal_stat, kruskal_p = stats.kruskal(d1_items, d2_items, d3_items)

    # ── Interpretación del IEM ────────────────────────────────────────────────
    interpretacion_iem = _interpretar_iem(iem, direccion_mitigacion)

    # ── Interpretación del test de Kruskal-Wallis ─────────────────────────────
    if kruskal_p < 0.05:
        interpretacion_kruskal = (
            "Se rechaza H0: las distribuciones de reducción difieren "
            "significativamente entre dominios (p < 0.05)"
        )
    else:
        interpretacion_kruskal = (
            "No se rechaza H0: las distribuciones son estadísticamente "
            "equivalentes entre dominios (p >= 0.05)"
        )

    return {
        "mu_D1":                 mu_D1,
        "mu_D2":                 mu_D2,
        "mu_D3":                 mu_D3,
        "mu_global":             mu_global,
        "sigma_inter":           sigma_inter,
        "IEM":                   iem,
        "iem_robusto":           iem_robusto,
        "n_excluidos_robustez":  n_excluidos_robustez,
        "direccion_mitigacion":  direccion_mitigacion,
        "kruskal_H":             float(kruskal_stat),
        "kruskal_p":             float(kruskal_p),
        "interpretacion_IEM":    interpretacion_iem,
        "interpretacion_kruskal": interpretacion_kruskal,
    }
