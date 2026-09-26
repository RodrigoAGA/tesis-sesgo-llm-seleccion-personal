"""
pipeline/metrics.py
===================
Funciones de cálculo de métricas para el experimento de sesgo lingüístico.

Funciones:
  - compute_reduction_pct:  porcentaje de reducción del sesgo entre condiciones
  - build_results_dataframe: construye el DataFrame principal de resultados
  - print_summary_table:    imprime resumen tabular en consola

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import pandas as pd


def compute_reduction_pct(delta_base, delta_mitigado):
    """
    Calcula el porcentaje de reducción del sesgo entre la condición control
    (delta_base) y la condición experimental (delta_mitigado).

    Fórmula:
        reduccion_pct = (|delta_base| - |delta_mitigado|) / |delta_base| * 100

    Un valor positivo indica reducción del sesgo (la mitigación funcionó).
    Un valor negativo indicaría que el PET amplificó el sesgo.
    Un valor de 100 indica eliminación completa del sesgo.

    Parámetros
    ----------
    delta_base : float
        Delta de sesgo en condición control (sin PET).
    delta_mitigado : float
        Delta de sesgo en condición experimental (con PET).

    Retorna
    -------
    float
        Porcentaje de reducción del sesgo. Retorna 0.0 si delta_base es cero
        (sin sesgo base no hay nada que reducir).
    """
    # Evitar división por cero cuando el modelo no muestra sesgo base
    if delta_base == 0.0:
        return 0.0

    reduccion = (abs(delta_base) - abs(delta_mitigado)) / abs(delta_base) * 100.0
    return reduccion


def build_results_dataframe(results_list):
    """
    Construye el DataFrame principal de resultados del experimento.

    Recibe la lista de diccionarios generada durante la ejecución del pipeline
    y añade la columna de reducción porcentual calculada.

    Parámetros
    ----------
    results_list : list[dict]
        Lista de diccionarios con las siguientes claves obligatorias:
          id, item_peninsular, item_latam, dominio, categoria,
          respuesta_control_pen, respuesta_control_lat,
          respuesta_experimental_pen, respuesta_experimental_lat,
          delta_base, delta_mitigado

    Retorna
    -------
    pandas.DataFrame
        DataFrame con todas las columnas originales más 'reduccion_pct'.
    """
    # Construir el DataFrame con el orden canónico de columnas
    columnas = [
        "id",
        "item_peninsular",
        "item_latam",
        "dominio",
        "categoria",
        "respuesta_control_pen",
        "respuesta_control_lat",
        "respuesta_experimental_pen",
        "respuesta_experimental_lat",
        "delta_base",
        "delta_mitigado",
    ]

    df = pd.DataFrame(results_list, columns=columnas)

    # Calcular la reducción porcentual para cada ítem
    df["reduccion_pct"] = df.apply(
        lambda fila: compute_reduction_pct(fila["delta_base"], fila["delta_mitigado"]),
        axis=1
    )

    return df


def print_summary_table(df):
    """
    Imprime en consola una tabla resumen con los resultados principales
    de cada ítem y la reducción global promedio al final.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de resultados generado por build_results_dataframe().
    """
    # Cabecera de la tabla
    separador = "-" * 70
    print("\n" + separador)
    print(
        "{:<5} {:<6} {:<12} {:<14} {:<14}".format(
            "ID", "Dom.", "Delta Base", "Delta Mitig.", "Reducción %"
        )
    )
    print(separador)

    # Imprimir una fila por ítem
    for _, fila in df.iterrows():
        print(
            "{:<5} {:<6} {:<12.4f} {:<14.4f} {:<14.2f}".format(
                int(fila["id"]),
                fila["dominio"],
                fila["delta_base"],
                fila["delta_mitigado"],
                fila["reduccion_pct"],
            )
        )

    # Resumen global al final
    print(separador)
    reduccion_global = df["reduccion_pct"].mean()
    print(
        "Reducción global promedio: " + str(round(reduccion_global, 2)) + "%"
        + "  (sobre " + str(len(df)) + " ítems procesados)"
    )
    print(separador + "\n")
