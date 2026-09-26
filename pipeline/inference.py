"""
pipeline/inference.py
=====================
Clase LlamaInference: cliente para la API de Groq que ejecuta el modelo
Llama-3-8B-Instruct en condición control (sin PET) y experimental (con PET).

Características:
  - Carga la API key desde el archivo .env usando python-dotenv
  - Soporta retry automático (hasta 3 intentos con 2 s de espera)
  - Expone métodos para generar respuestas en ambas condiciones experimentales

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import time
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
import groq

from corpus.corpus_v2 import SYSTEM_PROMPT_PET

# Nota: llama3-8b-8192 fue retirado (decommissioned) en junio 2025, y su
# sucesor llama-3.1-8b-instant también dejó de estar disponible en la API de
# Groq (verificado 2026-09-04: ya no aparece en client.models.list(), ningún
# modelo de la familia Llama 3 sigue vigente en el catálogo de Groq).
# Sustituido por un modelo open-weight comparable en espíritu (peso abierto,
# no propietario cerrado) todavía disponible: openai/gpt-oss-20b.
# NOTA: gpt-oss-20b es un modelo de razonamiento (piensa en cadena de
# pensamiento antes de responder). Con max_tokens=200 y sin ajustar el
# esfuerzo de razonamiento, ~40% de las respuestas volvían vacías
# (finish_reason="length", todo el presupuesto de tokens consumido en
# razonamiento interno, cero tokens para la respuesta final). Se agregó
# reasoning_effort="low" en _call_api para dejar tokens suficientes para
# la respuesta real (verificado 2026-09-04).
GROQ_MODEL = "openai/gpt-oss-20b"

# Número máximo de reintentos ante fallos de API
MAX_RETRIES = 3

# Segundos de espera entre reintentos
RETRY_WAIT = 2


class LlamaInference:
    """
    Cliente de inferencia para el modelo Llama-3-8B-Instruct vía Groq API.

    Permite generar respuestas en dos condiciones:
      - Control:      sin prompt de sistema (el modelo responde sin instrucción de mitigación)
      - Experimental: con Prompt de Estabilización Terminológica (PET) como system message
    """

    def __init__(self):
        """
        Inicializa el cliente Groq cargando la API key desde el archivo .env.

        Lanza ValueError si la clave no está configurada, con instrucciones claras
        para resolver el problema.
        """
        # Cargar variables de entorno desde el archivo .env del directorio actual
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        # Validar que la key existe y no es el valor de ejemplo
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "\n[ERROR] GROQ_API_KEY no configurada.\n"
                "Pasos para solucionarlo:\n"
                "  1. Copia el archivo .env.example a .env:  cp .env.example .env\n"
                "  2. Edita .env y reemplaza 'your_groq_api_key_here' con tu API key real.\n"
                "  3. Obtén una API key gratuita en: https://console.groq.com\n"
            )

        # Crear el cliente oficial de Groq
        self.client = groq.Groq(api_key=api_key)
        print("[LlamaInference] Cliente Groq inicializado correctamente.")
        print("[LlamaInference] Modelo: " + GROQ_MODEL + "  (Llama-3.1-8B-Instruct)")

    def _call_api(self, messages, temperature=0.0, max_tokens=200):
        """
        Realiza una llamada a la API de Groq con manejo de errores y reintentos.

        Parámetros
        ----------
        messages : list[dict]
            Lista de mensajes en formato OpenAI (role + content).
        temperature : float
            Temperatura de muestreo (0.0 para reproducibilidad máxima).
        max_tokens : int
            Número máximo de tokens en la respuesta.

        Retorna
        -------
        str | None
            Texto de la respuesta del modelo, o None si los 3 intentos fallaron.
        """
        for intento in range(1, MAX_RETRIES + 1):
            try:
                respuesta = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort="low",
                )
                return respuesta.choices[0].message.content

            except Exception as e:
                print(
                    "[LlamaInference] Intento " + str(intento) + "/" + str(MAX_RETRIES)
                    + " fallido: " + str(e)
                )
                if intento < MAX_RETRIES:
                    print(
                        "[LlamaInference] Reintentando en "
                        + str(RETRY_WAIT) + " segundos..."
                    )
                    time.sleep(RETRY_WAIT)
                else:
                    print("[LlamaInference] Se agotaron los reintentos. Retornando None.")
                    return None

    def generate_control(self, stimulus_text, item_id=None):
        """
        Genera una respuesta en condición CONTROL (sin prompt de sistema).

        En esta condición el LLM responde libremente al estímulo, sin ninguna
        instrucción de mitigación. Esto mide el sesgo base del modelo.

        Parámetros
        ----------
        stimulus_text : str
            Texto del estímulo a procesar.
        item_id : int | None
            ID del ítem del corpus (para logging).

        Retorna
        -------
        str | None
            Texto de la respuesta del modelo.
        """
        # En condición control solo hay un mensaje de usuario; sin system prompt
        messages = [
            {"role": "user", "content": stimulus_text}
        ]

        prefijo = "[ID " + str(item_id) + "]" if item_id is not None else ""
        resultado = self._call_api(messages)

        estado = "OK" if resultado is not None else "FALLO"
        print("  " + prefijo + " Control    -> " + estado)

        return resultado

    def generate_experimental(self, stimulus_text, item_id=None):
        """
        Genera una respuesta en condición EXPERIMENTAL (con PET como system message).

        En esta condición el LLM recibe el Prompt de Estabilización Terminológica
        que lo instruye a actuar como mediador sociolingüístico neutral.

        Parámetros
        ----------
        stimulus_text : str
            Texto del estímulo a procesar.
        item_id : int | None
            ID del ítem del corpus (para logging).

        Retorna
        -------
        str | None
            Texto de la respuesta del modelo.
        """
        # En condición experimental: system = PET + user = estímulo
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PET},
            {"role": "user",   "content": stimulus_text},
        ]

        prefijo = "[ID " + str(item_id) + "]" if item_id is not None else ""
        resultado = self._call_api(messages)

        estado = "OK" if resultado is not None else "FALLO"
        print("  " + prefijo + " Experimental -> " + estado)

        return resultado

    def generate_both(self, stimulus_text, item_id=None):
        """
        Genera respuestas en ambas condiciones (control y experimental).

        Parámetros
        ----------
        stimulus_text : str
            Texto del estímulo a procesar.
        item_id : int | None
            ID del ítem del corpus (para logging).

        Retorna
        -------
        dict
            Diccionario con claves 'control' y 'experimental' conteniendo
            los textos de respuesta (o None si la llamada falló).
        """
        return {
            "control":      self.generate_control(stimulus_text, item_id=item_id),
            "experimental": self.generate_experimental(stimulus_text, item_id=item_id),
        }
