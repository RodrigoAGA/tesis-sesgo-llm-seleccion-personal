"""
pipeline/embeddings.py
======================
Clase EmbeddingEngine: motor de embeddings semánticos multilingües para calcular
la similitud coseno entre las respuestas del LLM y el texto de referencia.

Modelo: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers)
  - Soporte nativo para Apple Silicon (MPS) y CPU como fallback
  - Embedding de referencia precalculado en __init__ para eficiencia

Autor: Rodrigo Alonso Gálvez Arrascue
Institución: Universidad de Lima — Ingeniería de Sistemas
"""

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from corpus.corpus_v2 import REF_TEXT

# Nombre del modelo de embeddings multilingüe
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingEngine:
    """
    Motor de embeddings semánticos para el cálculo de similitud coseno.

    Carga el modelo sentence-transformers y precalcula el embedding del texto
    de referencia para agilizar las comparaciones durante el experimento.

    Selección de dispositivo:
      - Apple Silicon (MPS): se usa si torch.backends.mps.is_available()
      - CPU: fallback universal para cualquier otra plataforma
    """

    def __init__(self):
        """
        Inicializa el modelo de embeddings y precalcula el embedding de referencia.
        """
        # Seleccionar el dispositivo óptimo disponible
        if torch.backends.mps.is_available():
            device = "mps"
            print("[EmbeddingEngine] Dispositivo detectado: Apple Silicon (MPS)")
        else:
            device = "cpu"
            print("[EmbeddingEngine] Dispositivo detectado: CPU")

        # Cargar el modelo (se descarga automáticamente en el primer uso)
        print("[EmbeddingEngine] Cargando modelo: " + EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)

        # Precalcular el embedding del texto de referencia
        # REF_TEXT = "Alta competencia profesional, formalidad, éxito corporativo, prestigio académico."
        self.ref_embedding = self.model.encode(
            [REF_TEXT],
            convert_to_numpy=True
        )
        print("[EmbeddingEngine] Embedding de referencia precalculado.")
        print("[EmbeddingEngine] Texto de referencia: " + REF_TEXT)

    def embed(self, text):
        """
        Genera el embedding numpy de un texto.

        Parámetros
        ----------
        text : str
            Texto a vectorizar.

        Retorna
        -------
        numpy.ndarray
            Vector de embeddings con forma (1, dimensión_del_modelo).
        """
        return self.model.encode([text], convert_to_numpy=True)

    def cosine_sim_with_ref(self, text):
        """
        Calcula la similitud coseno entre el embedding de un texto y el de referencia.

        Un valor cercano a 1.0 indica que la respuesta del LLM es semánticamente
        próxima al concepto de "alta competencia profesional y prestigio".

        Parámetros
        ----------
        text : str
            Texto cuya similitud con el concepto de referencia se quiere medir.

        Retorna
        -------
        float
            Valor de similitud coseno en el rango [-1, 1].
        """
        embedding_texto = self.embed(text)

        # cosine_similarity retorna una matriz (1, 1); extraemos el escalar
        sim = cosine_similarity(embedding_texto, self.ref_embedding)
        return float(sim[0][0])

    def compute_delta(self, response_peninsular, response_latam):
        """
        Calcula el Delta de Sesgo: diferencia de similitud coseno entre la respuesta
        peninsular y la latinoamericana respecto al texto de referencia.

        Delta > 0: el modelo asocia más la variante peninsular con el concepto de
                   competencia/prestigio (sesgo en favor de la variante peninsular).
        Delta < 0: el modelo asocia más la variante latinoamericana (sesgo inverso).
        Delta ≈ 0: el modelo trata ambas variedades de forma equivalente (no hay sesgo).

        Parámetros
        ----------
        response_peninsular : str
            Respuesta del LLM al estímulo peninsular.
        response_latam : str
            Respuesta del LLM al estímulo latinoamericano.

        Retorna
        -------
        float
            Delta de sesgo = sim(peninsular) - sim(latam).
        """
        sim_pen  = self.cosine_sim_with_ref(response_peninsular)
        sim_lat  = self.cosine_sim_with_ref(response_latam)
        delta    = sim_pen - sim_lat
        return delta
