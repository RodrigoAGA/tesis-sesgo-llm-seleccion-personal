"""Genera las figuras (PNG) y las tablas del informe a partir de resultados/<prefijo>_tablas.json y los .jsonl crudos.

Uso: python -m auditoria.informe --prefijo completa --salida /ruta/figuras
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analisis as A
from . import diseno as D

# Paleta Okabe-Ito (distinguible con daltonismo y en escala de grises)
COL = {"groq-gpt-oss-20b": "#0072B2", "groq-qwen3.8-27b": "#D55E00", "gemini-3.1-flash-lite": "#009E73", "gemini-3.5-flash-lite": "#CC79A7", "groq-gpt-oss-120b": "#E69F00"}
NOM = {"groq-gpt-oss-20b": "GPT-OSS-20B", "groq-qwen3.8-27b": "Qwen3.8-27B", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "groq-gpt-oss-120b": "GPT-OSS-120B"}
ORDEN_FIG = ["groq-gpt-oss-20b", "groq-gpt-oss-120b", "groq-qwen3.8-27b", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def fig_forest(res, out):
    attrs = list(A.ATRIBUTOS)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    h = 0.16
    nm = len(res)
    for i, r in enumerate(res):
        ys, ms, lo, hi = [], [], [], []
        for j, a in enumerate(attrs):
            e = next((x for x in r["efectos"] if x["atributo"] == a), None)
            if e is None:
                continue
            ys.append(j + (i - (nm - 1) / 2) * h)
            ms.append(e["dif_media"]); lo.append(e["dif_media"] - e["ic95"][0]); hi.append(e["ic95"][1] - e["dif_media"])
        ax.errorbar(ms, ys, xerr=[lo, hi], fmt="o", color=COL[r["modelo"]], capsize=3, label=NOM[r["modelo"]], ms=4)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels([A.ATRIBUTOS[a] for a in attrs])
    ax.invert_yaxis()
    ax.set_xlabel("Diferencia media de puntaje frente a la referencia\n(puntos de 0 a 100; IC 95 % por bootstrap de CVs)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_efectos.png")
    plt.close(fig)


def fig_monotonia(df, out):
    nmod = df.modelo.nunique()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    orden = ["calidad_baja", "ref", "calidad_alta"]
    etq = ["Baja", "Media (referencia)", "Alta"]
    w = 0.16
    base = df[(df.tipo == "puntaje") & (df.valido == True) & (df.rep == 1)]  # noqa: E712
    modelos = [m for m in ORDEN_FIG if m in set(base.modelo)]
    nmod = len(modelos)
    for i, m in enumerate(modelos):
        g = base[base.modelo == m]
        medias = [g[g.variante == v].puntaje.mean() for v in orden]
        ds = [g[g.variante == v].puntaje.std() for v in orden]
        ax.bar(np.arange(3) + (i - (nmod - 1) / 2) * w, medias, w, yerr=ds, color=COL[m], label=NOM[m], capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(etq)
    ax.set_ylabel("Puntaje medio (0 a 100)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    fig.tight_layout(); fig.savefig(out / "fig_monotonia.png"); plt.close(fig)


def fig_dir(res, out):
    attrs = list(A.ATRIBUTOS)
    fig, axs = plt.subplots(1, len(attrs), figsize=(7.6, 2.6), sharey=True)
    for ax, a in zip(np.atleast_1d(axs), attrs):
        for r in res:
            e = next((x for x in r["efectos"] if x["atributo"] == a), None)
            if not e:
                continue
            us = [x["umbral"] for x in e["por_umbral"]]
            ds = [x["dir"] if x["informativo"] else np.nan for x in e["por_umbral"]]
            ax.plot(us, ds, "-o", ms=3, color=COL[r["modelo"]], label=NOM[r["modelo"]])
        ax.axhline(0.8, color="#B00020", ls="--", lw=0.8)
        ax.set_title(A.ATRIBUTOS[a], fontsize=8)
        ax.set_xlabel("Umbral de puntaje", fontsize=8)
        ax.set_ylim(0, 1.05)
    np.atleast_1d(axs)[0].set_ylabel("Razón de impacto dispar")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig_dir.png"); plt.close(fig)


def fig_orden(res, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    vs = ["edad60", "mujer", "ayacucho", "caracas"]
    w = 0.18
    con = [r for r in res if r.get("orden")]
    for i, r in enumerate(con):
        o = r["orden"]
        vals = [o["por_variante"].get(v, {}).get("p_elige_ref", np.nan) for v in vs]
        ax.bar(np.arange(len(vs)) + (i - (len(con) - 1) / 2) * w, vals, w, color=COL[r["modelo"]], label=NOM[r["modelo"]])
    ax.axhline(0.5, color="#444", lw=0.8)
    ax.set_xticks(range(len(vs))); ax.set_xticklabels([A.ATRIBUTOS[v] for v in vs], fontsize=7)
    ax.set_ylabel("Proporción en que se elige la referencia"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4)
    fig.tight_layout(); fig.savefig(out / "fig_orden.png"); plt.close(fig)


def fig_arquitectura(out):
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    textos = ["Avisos reales\n(Computrabajo)", "Generador de\nCVs base\n(con semilla)", "Variantes por\natributo y\ncalidad", "Conector API\n(Groq, Gemini)",
              "Batería R1 a R4\nrespuesta JSON", "Análisis y\ncriterio de tres\nestados"]
    w, gap = 0.135, 0.028
    for i, t in enumerate(textos):
        x = 0.01 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.38), w, 0.40, fill=False, lw=1.2, ec="#222"))
        ax.text(x + w / 2, 0.58, t, ha="center", va="center", fontsize=6.6)
        if i < len(textos) - 1:
            ax.annotate("", xy=(x + w + gap, 0.58), xytext=(x + w, 0.58), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(0.5, 0.15, "Salida: resultados crudos (.jsonl), tablas estadísticas y figuras reproducibles", ha="center", fontsize=7, style="italic")
    fig.savefig(out / "fig_arquitectura.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefijo", default="completa")
    ap.add_argument("--salida", default=str(D.RAIZ / "figuras"))
    a = ap.parse_args()
    out = Path(a.salida); out.mkdir(parents=True, exist_ok=True)
    res = json.loads((A.RES / f"{a.prefijo}_tablas.json").read_text())
    res = [r for r in res if r["modelo"] in COL]
    orden = ["groq-gpt-oss-20b", "groq-gpt-oss-120b", "groq-qwen3.8-27b", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    res = [r for r in res if r["efectos"]]
    res.sort(key=lambda r: orden.index(r["modelo"]))
    df = A.cargar(a.prefijo)
    fig_forest(res, out); fig_monotonia(df, out); fig_dir(res, out); fig_orden(res, out); fig_arquitectura(out)
    print("figuras en", out)


if __name__ == "__main__":
    main()
