"""Embeddings via the Gemini API (hosted, so the deployed app stays small)."""
import time

from google import genai
from google.genai import types

from .config import settings

BATCH_SIZE = 50
_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _with_retry(fn, attempts: int = 4):
    delay = 1.5
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = _with_retry(
            lambda: client.models.embed_content(
                model=settings.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dim,
                ),
            )
        )
        vectors.extend(list(e.values) for e in result.embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    return embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
