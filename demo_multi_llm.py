"""
demo_multi_llm.py
==================
Prueba de concepto del conector multi-LLM (pipeline/llm_connector.py):
manda el mismo prompt a distintos proveedores usando la misma interfaz
(LLMProvider.generate) y muestra las respuestas lado a lado.

Este script NO es parte de la batería de tests de sesgo en RRHH todavía
(esa se diseña aparte) — es solo la validación de que el conector
funciona contra más de un LLM real.

Uso:
    python demo_multi_llm.py

Prerrequisito:
    .env con GROQ_API_KEY y GEMINI_API_KEY (ver .env.example)
"""

from dotenv import load_dotenv

from pipeline.llm_connector import get_provider

load_dotenv()

PROMPT_DE_PRUEBA = (
    "Evalúa en una frase la idoneidad de este candidato para un puesto de "
    "atención al cliente: 'Tengo 3 años de experiencia en soporte técnico "
    "y buena comunicación.'"
)

if __name__ == "__main__":
    for provider_name in ["groq", "gemini"]:
        print("=" * 64)
        print(f"Proveedor: {provider_name}")
        print("=" * 64)
        try:
            provider = get_provider(provider_name)
            respuesta = provider.generate([{"role": "user", "content": PROMPT_DE_PRUEBA}])
            print(respuesta or "[sin respuesta]")
        except Exception as e:
            print(f"[ERROR] {e}")
        print()
