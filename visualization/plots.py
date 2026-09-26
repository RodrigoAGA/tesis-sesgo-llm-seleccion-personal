"""
visualization/plots.py
======================
Funciones de visualización académica para el experimento de sesgo lingüístico.

Estilo consistente en todas las figuras:
  - Fondo blanco, fuente DejaVu Sans, dpi=300
  - Color control:      #C0392B (rojo oscuro)
  - Color experimental: #27AE60 (verde oscuro)
  - Sin decoraciones innecesarias (minimalist academic style)

Funciones:
  - plot_delta_by_domain     : barras agrupadas Delta Base vs. Delta Mitigado
  - plot_iem_consistency     : barras de reducción promedio por dominio con IEM
  - plot_reduction_heatmap   : mapa de calor seaborn por ítem y dominio

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

# Usar backend no interactivo para evitar problemas en entornos sin display
matplotlib.use("Agg")

# ── Paleta de colores global ──────────────────────────────────────────────────
COLOR_CONTROL      = "#C0392B"   # rojo oscuro — condición control (sin PET)
COLOR_EXPERIMENTAL = "#27AE60"   # verde oscuro — condición experimental (con PET)

# Colores de fondo para cada dominio en el gráfico de barras
COLOR_FONDO_D1 = "#D6EAF8"       # azul claro
COLOR_FONDO_D2 = "#FADBD8"       # rojo claro
COLOR_FONDO_D3 = "#D5F5E3"       # verde claro

# Configuración de fuente académica
FONT_FAMILY = "DejaVu Sans"
DPI         = 300


def _setup_estilo_academico():
    """Aplica el estilo académico base a matplotlib."""
    plt.rcParams.update({
        "font.family":      FONT_FAMILY,
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
    })


def plot_delta_by_domain(df, output_path):
    """
    Gráfico de barras agrupadas: Delta Base (control) vs. Delta Mitigado (experimental)
    para cada ítem del corpus, con fondos de color por dominio semántico.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de resultados con columnas: id, dominio, delta_base, delta_mitigado.
    output_path : str
        Ruta absoluta o relativa donde se guardará el archivo PNG.
    """
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _setup_estilo_academico()

    n_items  = len(df)
    ids      = df["id"].tolist()
    dominios = df["dominio"].tolist()

    # Posiciones de las barras en el eje X
    x        = np.arange(n_items)
    ancho    = 0.35  # ancho de cada barra

    fig, ax = plt.subplots(figsize=(18, 7))

    # ── Dibujar barras agrupadas ──────────────────────────────────────────────
    barras_base = ax.bar(
        x - ancho / 2,
        df["delta_base"].tolist(),
        ancho,
        label="Delta Base (Control — sin PET)",
        color=COLOR_CONTROL,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    barras_mit = ax.bar(
        x + ancho / 2,
        df["delta_mitigado"].tolist(),
        ancho,
        label="Delta Mitigado (Experimental — con PET)",
        color=COLOR_EXPERIMENTAL,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    # ── Fondos de color por dominio ───────────────────────────────────────────
    # Identificar los límites de cada dominio (grupo de ítems consecutivos)
    grupos_dominio = {}
    for idx, (item_id, dominio) in enumerate(zip(ids, dominios)):
        if dominio not in grupos_dominio:
            grupos_dominio[dominio] = []
        grupos_dominio[dominio].append(idx)

    colores_fondo = {"D1": COLOR_FONDO_D1, "D2": COLOR_FONDO_D2, "D3": COLOR_FONDO_D3}
    etiquetas_dom = {
        "D1": "D1 — Léxico Coloquial",
        "D2": "D2 — Terminología Profesional",
        "D3": "D3 — Morfosintaxis Formal",
    }

    # Calcular rango Y para dimensionar los rectángulos de fondo
    todos_deltas = df["delta_base"].tolist() + df["delta_mitigado"].tolist()
    y_min = min(todos_deltas) - 0.02
    y_max = max(todos_deltas) + 0.04

    for dominio, indices in grupos_dominio.items():
        inicio = min(indices) - 0.5
        fin    = max(indices) + 0.5
        ancho_rect = fin - inicio

        # Rectángulo de fondo semitransparente
        rect = mpatches.FancyBboxPatch(
            (inicio, y_min),
            ancho_rect,
            y_max - y_min,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=colores_fondo[dominio],
            alpha=0.35,
            zorder=0,
        )
        ax.add_patch(rect)

        # Etiqueta del dominio encima del grupo
        centro = (inicio + fin) / 2
        ax.text(
            centro,
            y_max - 0.005,
            etiquetas_dom[dominio],
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2C3E50",
        )

    # ── Línea horizontal en y=0 ───────────────────────────────────────────────
    ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-", alpha=0.8)

    # ── Etiquetas y decoración ────────────────────────────────────────────────
    ax.set_xlabel("ID del Ítem", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        "Delta de Sesgo (Diferencia de Similitud Coseno)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "Análisis de Bidireccionalidad del Sesgo por Dominio Semántico — "
        "Corpus Hispano-Bias v2.0",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], fontsize=8)
    ax.set_ylim(y_min, y_max + 0.01)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[Plots] Gráfico guardado: " + output_path)


def plot_iem_consistency(iem_results, output_path):
    """
    Gráfico de barras simples que muestra la reducción promedio del sesgo por dominio
    y la media global como línea punteada, junto con el IEM y su interpretación.

    Parámetros
    ----------
    iem_results : dict
        Diccionario retornado por compute_iem(), con claves mu_D1, mu_D2, mu_D3,
        mu_global, IEM e interpretacion_IEM.
    output_path : str
        Ruta absoluta o relativa donde se guardará el archivo PNG.
    """
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _setup_estilo_academico()

    # Datos a visualizar
    etiquetas = [
        "D1 — Léxico\nColoquial",
        "D2 — Terminología\nProfesional",
        "D3 — Morfosintaxis\nFormal",
    ]
    valores = [
        iem_results["mu_D1"],
        iem_results["mu_D2"],
        iem_results["mu_D3"],
    ]
    colores = ["#2E86C1", "#C0392B", "#1E8449"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Barras por dominio ────────────────────────────────────────────────────
    x = np.arange(len(etiquetas))
    barras = ax.bar(
        x,
        valores,
        width=0.5,
        color=colores,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    # ── Etiquetas de valor sobre cada barra ───────────────────────────────────
    for barra, valor in zip(barras, valores):
        altura = barra.get_height()
        # Manejar valores negativos (barra hacia abajo)
        offset = 1.0 if altura >= 0 else -3.5
        ax.text(
            barra.get_x() + barra.get_width() / 2.0,
            altura + offset,
            str(round(valor, 2)) + "%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#2C3E50",
        )

    # ── Línea del promedio global ─────────────────────────────────────────────
    mu_global = iem_results["mu_global"]
    ax.axhline(
        y=mu_global,
        color="black",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Promedio Global = " + str(round(mu_global, 2)) + "%",
    )

    # ── Título principal y subtítulo con IEM + dirección del efecto ──────────────
    valor_iem  = iem_results["IEM"]
    interp_iem = iem_results["interpretacion_IEM"]
    direccion  = iem_results.get("direccion_mitigacion", "")

    # Etiqueta de dirección para el subtítulo (capitalizada)
    dir_label = direccion.capitalize() if direccion else ""

    ax.set_title(
        "Consistencia de Mitigación entre Dominios Semánticos",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    # Subtítulo: valor IEM + dirección del efecto global
    fig.text(
        0.5,
        0.93,
        "IEM = " + str(round(valor_iem, 4)) + " — " + dir_label + " del sesgo",
        ha="center",
        va="center",
        fontsize=10,
        color="#5D6D7E",
        style="italic",
    )
    # Nota al pie metodológica
    fig.text(
        0.5,
        0.01,
        "IEM corregido con |μ_global|. Análisis de robustez excluye ítems con |Δbase| < 0.01.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#7F8C8D",
        style="italic",
    )

    # ── Decoración ────────────────────────────────────────────────────────────
    ax.set_ylabel("Reducción promedio del sesgo (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=11)
    # Ajustar ylim dinámicamente para acomodar barras negativas
    y_min_plot = min(min(valores) - 20, -20) if min(valores) < 0 else -5
    y_max_plot = max(max(valores) + 20, 20)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # subplots_adjust evita el UserWarning de tight_layout cuando se usa fig.text
    # (tight_layout no puede acomodar texto posicionado con coordenadas de figura)
    plt.subplots_adjust(top=0.88, bottom=0.2, left=0.12, right=0.95)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[Plots] Gráfico IEM guardado: " + output_path)


def plot_reduction_heatmap(df, output_path):
    """
    Mapa de calor (heatmap) de la reducción porcentual del sesgo por ítem y dominio.

    Permite identificar visualmente qué ítems y dominios responden mejor o peor
    a la estrategia de mitigación PET.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de resultados con columnas: id, dominio, reduccion_pct.
    output_path : str
        Ruta absoluta o relativa donde se guardará el archivo PNG.
    """
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _setup_estilo_academico()

    # ── Crear tabla pivote: filas = ítems, columnas = dominios ───────────────
    # Para el heatmap necesitamos un formato de matriz
    tabla_pivot = df.pivot_table(
        index="id",
        columns="dominio",
        values="reduccion_pct",
        aggfunc="first",
    )

    # Asegurar orden de dominios
    columnas_ordenadas = [c for c in ["D1", "D2", "D3"] if c in tabla_pivot.columns]
    tabla_pivot = tabla_pivot[columnas_ordenadas]

    fig, ax = plt.subplots(
        figsize=(max(6, len(columnas_ordenadas) * 2.5), max(8, len(tabla_pivot) * 0.55))
    )

    sns.heatmap(
        tabla_pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="#ECEFF1",
        ax=ax,
        cbar_kws={"label": "Reducción del sesgo (%)"},
        annot_kws={"size": 9},
    )

    ax.set_title(
        "Mapa de Calor de Reducción del Sesgo por Ítem y Dominio",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Dominio Semántico", fontsize=11, fontweight="bold")
    ax.set_ylabel("ID del Ítem", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[Plots] Heatmap guardado: " + output_path)
