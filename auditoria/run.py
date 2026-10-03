"""Corre la batería. Reanudable: cada fila va a un .jsonl y se salta lo ya hecho.

Uso:
  python -m auditoria.run --fase piloto --modelos groq-gpt-oss-20b gemini-2.5-flash
  python -m auditoria.run --fase completa --modelos ... --reps 3 --bases 6
Fases:
  piloto    2 ocupaciones x 3 CVs base, redacción p1, 1 repetición, solo puntaje (+ pares)
  completa  2 ocupaciones x N bases, 3 redacciones, R repeticiones, + pares ambos órdenes
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import diseno as D
from .llm import MODELOS

SALIDA = D.RAIZ / "resultados"


def extraer_json(txt):
    if not txt or txt.startswith("__ERROR__"):
        return None
    m = re.search(r"\{.*?\}", txt, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def pruebas(fase, n_bases, redacciones, reps, temperatura):
    avisos = D.cargar_avisos()
    out = []
    for oc in D.OCUPACIONES:
        aviso = avisos[oc]["texto"]
        for i in range(1, n_bases + 1):
            base = D.generar_base(oc, i)
            for red in redacciones:
                for var in D.VARIANTES:
                    for rep in range(1, reps + 1):
                        out.append(dict(tipo="puntaje", oc=oc, base=i, variante=var, redaccion=red, rep=rep, temp=temperatura,
                                        mensajes=D.construir_mensajes(red, aviso, D.renderizar_cv(base, var))))
            # orden: referencia vs cada variante de atributo, en ambos órdenes
            for var in ("edad60", "mujer", "ayacucho", "caracas", "calidad_alta", "calidad_baja"):
                for orden in ("ref_primero", "ref_segundo"):
                    cv_ref, cv_var = D.renderizar_cv(base, "ref"), D.renderizar_cv(base, var)
                    c1, c2 = (cv_ref, cv_var) if orden == "ref_primero" else (cv_var, cv_ref)
                    for rep in range(1, reps + 1):
                        out.append(dict(tipo="par", oc=oc, base=i, variante=var, redaccion="par", orden=orden, rep=rep, temp=temperatura,
                                        mensajes=D.construir_mensajes_par(aviso, c1, c2)))
    return out


def clave(p, modelo):
    return "|".join(str(x) for x in (modelo, p["tipo"], p["oc"], p["base"], p["variante"], p["redaccion"], p.get("orden", ""), p["rep"], p["temp"]))


def correr_modelo(nombre, plan, archivo, lock, hilos=1):
    hechos = set()
    if archivo.exists():
        for l in archivo.read_text().splitlines():
            try:
                hechos.add(json.loads(l)["clave"])
            except Exception:
                pass
    prov = MODELOS[nombre]()
    pend = [p for p in plan if clave(p, nombre) not in hechos]
    print(f"[{nombre}] pendientes {len(pend)} de {len(plan)}", flush=True)
    n = [0]
    def una(p):
        txt = prov.generar(p["mensajes"], temperature=p["temp"], max_tokens=300)
        js = extraer_json(txt)
        if txt is not None and txt.startswith("__ERROR__"):
            # fallo de infraestructura (cuota, red): no es una respuesta del modelo; no se registra y se reintenta en la próxima corrida
            with lock:
                (SALIDA / "errores_api.log").open("a").write(f"{datetime.now(timezone.utc).isoformat()} {nombre} {clave(p, nombre)} {txt[:160]!r}\n")
            return
        fila = {k: v for k, v in p.items() if k != "mensajes"}
        fila.update(clave=clave(p, nombre), modelo=nombre, version=prov.modelo, ts=datetime.now(timezone.utc).isoformat(),
                    crudo=(txt or "")[:400], valido=js is not None)
        if js:
            if p["tipo"] == "puntaje":
                try:
                    fila["puntaje"] = float(js.get("puntaje"))
                    fila["pre"] = bool(js.get("preseleccionado"))
                except Exception:
                    fila["valido"] = False
            else:
                fila["elegido"] = js.get("elegido")
        with lock, archivo.open("a") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"[{nombre}] {n[0]}/{len(pend)}", flush=True)

    with ThreadPoolExecutor(hilos) as ex:
        list(ex.map(una, pend))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["piloto", "completa"], default="piloto")
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS))
    ap.add_argument("--bases", type=int)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--sin-pares", action="store_true")
    ap.add_argument("--etiqueta")
    ap.add_argument("--solo-variantes", nargs="+")
    ap.add_argument("--puntaje-primero", action="store_true")
    ap.add_argument("--redacciones", nargs="+")
    ap.add_argument("--hilos", type=int, default=1)
    a = ap.parse_args()
    if a.fase == "piloto":
        plan = pruebas("piloto", a.bases or 3, ["p1"], a.reps or 1, a.temp)
    else:
        plan = pruebas("completa", a.bases or 6, ["p1", "p2", "p3"], a.reps or 3, a.temp)
    if a.solo_variantes:
        plan = [p for p in plan if p["tipo"] == "puntaje" and p["variante"] in a.solo_variantes]
    if a.redacciones:
        plan = [p for p in plan if p["redaccion"] in a.redacciones]
    if a.sin_pares:
        plan = [p for p in plan if p["tipo"] == "puntaje"]
    if a.puntaje_primero:
        plan.sort(key=lambda p: p["tipo"] != "puntaje")
    SALIDA.mkdir(exist_ok=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(len(a.modelos)) as ex:
        for m in a.modelos:
            ex.submit(correr_modelo, m, plan, SALIDA / f"{a.etiqueta or a.fase}_{m}.jsonl", lock, a.hilos)


if __name__ == "__main__":
    main()
