"""
pipeline/llm_connector.py
==========================
Capa de conexión multi-LLM: interfaz común (LLMProvider.generate) para
llamar a distintos proveedores con la misma firma, de modo que la misma
batería de tests pueda correr contra "cualquier LLM" sin reescribir la
lógica de orquestación por proveedor.

Proveedores implementados hoy: Groq (gratis), Gemini (free tier de
Google AI Studio). OpenAI y Claude quedan como adapters pendientes —
ninguno de los dos tiene tier gratuito de API (Claude Pro es la app de
chat, no incluye acceso a la API).

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import os
import time

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
import groq
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types as genai_types

MAX_RETRIES = 3
RETRY_WAIT = 2


class LLMProvider:
    """Interfaz común que debe implementar cada proveedor."""

    name = "base"

    def generate(self, messages, temperature=0.0, max_tokens=200):
        """
        messages : list[dict] en formato OpenAI (role: system|user, content: str)
        Retorna el texto de la respuesta, o None si los reintentos se agotaron.
        """
        raise NotImplementedError

    def _retry(self, call_fn):
        for intento in range(1, MAX_RETRIES + 1):
            try:
                return call_fn()
            except Exception as e:
                print(f"[{self.name}] Intento {intento}/{MAX_RETRIES} fallido: {e}")
                if intento < MAX_RETRIES:
                    print(f"[{self.name}] Reintentando en {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                else:
                    print(f"[{self.name}] Se agotaron los reintentos. Retornando None.")
                    return None


class GroqProvider(LLMProvider):
    # Ver pipeline/inference.py para el historial de por qué no es Llama 3
    # (toda la familia fue retirada del catálogo de Groq, verificado 2026-09-04).
    name = "groq"
    model = "openai/gpt-oss-20b"

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY no configurada en .env (ver .env.example). "
                "Key gratuita en https://console.groq.com"
            )
        self.client = groq.Groq(api_key=api_key)

    def generate(self, messages, temperature=0.0, max_tokens=200):
        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="low",
            )
            return resp.choices[0].message.content
        return self._retry(_call)


class GeminiProvider(LLMProvider):
    name = "gemini"
    model = "gemini-2.5-flash"

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no configurada en .env (ver .env.example). "
                "Key gratuita (free tier) en https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)

    def generate(self, messages, temperature=0.0, max_tokens=200):
        # google-genai no usa el formato de mensajes de OpenAI: el system
        # prompt va aparte (system_instruction), el resto se concatena en contents.
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_text = "\n".join(m["content"] for m in messages if m["role"] != "system")

        def _call():
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user_text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_text or None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text
        return self._retry(_call)


class OpenAIProvider(LLMProvider):
    """TODO: falta OPENAI_API_KEY propia (API de pago, sin free tier)."""
    name = "openai"

    def __init__(self):
        raise NotImplementedError(
            "OpenAIProvider pendiente: falta configurar OPENAI_API_KEY "
            "(sin tier gratuito, requiere billing en platform.openai.com)."
        )

    def generate(self, messages, temperature=0.0, max_tokens=200):
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    """TODO: Claude Pro es la app de chat, no incluye acceso a la API."""
    name = "claude"

    def __init__(self):
        raise NotImplementedError(
            "ClaudeProvider pendiente: falta configurar ANTHROPIC_API_KEY "
            "(Claude Pro no incluye acceso a la API, es billing aparte en console.anthropic.com)."
        )

    def generate(self, messages, temperature=0.0, max_tokens=200):
        raise NotImplementedError


PROVIDERS = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
}


def get_provider(name):
    """Instancia un provider por nombre. Ver PROVIDERS para las opciones."""
    if name not in PROVIDERS:
        raise ValueError(f"Provider desconocido: '{name}'. Opciones: {list(PROVIDERS)}")
    return PROVIDERS[name]()
