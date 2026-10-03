"""Conectores mínimos (solo biblioteca estándar) a Groq y Gemini por REST.

Interfaz común: `generar(messages, temperature, max_tokens) -> str | None`.
Las claves se leen de `.env` (GROQ_API_KEY, GEMINI_API_KEY) o del entorno.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def _cargar_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for linea in p.read_text().splitlines():
            if "=" in linea and not linea.strip().startswith("#"):
                k, v = linea.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_cargar_env()


def _post(url, headers, payload, timeout=90):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "User-Agent": "tesis-auditoria-sesgo/1.0 (python)", **headers}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Proveedor:
    nombre = "base"
    modelo = ""
    pausa = 0.0  # segundos entre llamadas, para respetar límites de tasa

    def generar(self, messages, temperature=0.7, max_tokens=300, reintentos=6):
        i = 0
        while i < reintentos:
            i += 1
            try:
                out = self._llamar(messages, temperature, max_tokens)
                time.sleep(self.pausa)
                return out
            except urllib.error.HTTPError as e:
                cuerpo = e.read().decode(errors="ignore")
                diaria = e.code == 429 and re.search(r"per ?day|PerDay|tokens per day|requests per day", cuerpo, re.I)
                if diaria:
                    # cuota diaria gratuita agotada: se espera a que se reinicie (no se pierde el ítem)
                    time.sleep(600)
                    i = max(1, i - 1)
                    continue
                espera = min(60, 4 * i) if e.code in (429, 500, 502, 503) else 0
                if not espera or i == reintentos:
                    return f"__ERROR__ HTTP {e.code} {cuerpo[:200]}"
                time.sleep(espera)
            except Exception as e:  # red, timeout
                if i == reintentos:
                    return f"__ERROR__ {type(e).__name__} {e}"
                time.sleep(3 * i)


class Groq(Proveedor):
    nombre = "groq"
    pausa = 9.0  # el límite gratuito de Groq es ~7-8 mil tokens/min (~6 llamadas/min)

    def __init__(self, modelo="openai/gpt-oss-20b"):
        self.modelo = modelo
        self.clave = os.environ["GROQ_API_KEY"]

    def _llamar(self, messages, temperature, max_tokens):
        d = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {self.clave}"},
            {"model": self.modelo, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, "reasoning_effort": "low"} if "gpt-oss" in self.modelo else
            {"model": self.modelo, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "reasoning_effort": "none"} if "qwen" in self.modelo else
            {"model": self.modelo, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        return d["choices"][0]["message"]["content"]


class Gemini(Proveedor):
    nombre = "gemini"
    pausa = 1.0

    def __init__(self, modelo="gemini-2.5-flash"):
        self.modelo = modelo
        self.clave = os.environ["GEMINI_API_KEY"]

    def _llamar(self, messages, temperature, max_tokens):
        sistema = "\n".join(m["content"] for m in messages if m["role"] == "system")
        usuario = "\n".join(m["content"] for m in messages if m["role"] != "system")
        cuerpo = {
            "contents": [{"role": "user", "parts": [{"text": usuario}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(max_tokens, 1024)},
        }
        if "gemma" not in self.modelo and self.modelo in ("gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"):
            cuerpo["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        if sistema:
            cuerpo["systemInstruction"] = {"parts": [{"text": sistema}]}
        d = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.modelo}:generateContent?key={self.clave}",
            {}, cuerpo,
        )
        return d["candidates"][0]["content"]["parts"][0]["text"]


class MockPrueba(Proveedor):
    """SOLO PARA PROBAR LA TUBERÍA (prefijo 'test'). Nunca se reporta como resultado."""
    nombre = "mock"
    modelo = "mock-TEST"

    def _llamar(self, messages, temperature, max_tokens):
        import random, re
        u = messages[-1]["content"]
        r = random.Random(hash(u) % 10**6 + int(temperature * 1000))
        if "Candidato 1" in u:
            return json.dumps({"elegido": r.choice([1, 2])})
        anios = int(re.search(r"Total: (\d+)", u).group(1)) if "Total:" in u else 0
        s = 35 + 6 * anios + 4 * u.count("\n- ") + r.gauss(0, 4) - (7 if "(60 años)" in u else 0)
        s = max(0, min(100, s))
        return json.dumps({"puntaje": round(s), "preseleccionado": s >= 70})


# Modelos del estudio (nombre corto -> constructor). Versión y fecha se registran en cada fila.
MODELOS = {
    "mock-TEST": lambda: MockPrueba(),
    "groq-gpt-oss-20b": lambda: Groq("openai/gpt-oss-20b"),
    "groq-gpt-oss-120b": lambda: Groq("openai/gpt-oss-120b"),
    "groq-qwen3.8-27b": lambda: Groq("qwen/qwen3.8-27b"),
    "gemini-3.1-flash-lite": lambda: Gemini("gemini-3.1-flash-lite"),
    "gemini-3.5-flash-lite": lambda: Gemini("gemini-3.5-flash-lite"),
    "gemini-3.6-flash": lambda: Gemini("gemini-3.6-flash"),
    "gemma-4-31b": lambda: Gemini("gemma-4-31b-it"),
}
