import logging
import time
from functools import lru_cache
from urllib.parse import urlparse
import numpy as np
from .config import (EMBED_API_KEY, EMBED_BASE_URL, EMBED_DIM, EMBED_DOC_PREFIX, EMBED_MODEL,
                     EMBED_QUERY_PREFIX, EMBED_SEND_DIMENSIONS)
from .usage import est_tokens, extract_usage, record

log = logging.getLogger(__name__)
BATCH = 32
RETRYABLE = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


class EmbeddingError(RuntimeError):
    pass


def describe_backend() -> str:
    return f"{EMBED_MODEL} via {urlparse(EMBED_BASE_URL).netloc}" if EMBED_BASE_URL else f"{EMBED_MODEL}, local"


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    log.info("loading embedding model %s (first use downloads it)", EMBED_MODEL)
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    log.info("model ready in %.1fs", time.perf_counter() - t0)
    return model


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI
    return OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "none", timeout=120)


def _retryable(e: Exception) -> bool:
    return type(e).__name__ in RETRYABLE or getattr(e, "status_code", 0) in (429, 500, 502, 503, 504)


def _local(texts: list[str]) -> np.ndarray:
    return _local_model().encode(texts, normalize_embeddings=True, batch_size=16,
                                 show_progress_bar=False).astype("float32")


def _remote(texts: list[str]) -> np.ndarray:
    vectors = []
    host = urlparse(EMBED_BASE_URL).netloc
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        kwargs = {"model": EMBED_MODEL, "input": batch}
        if EMBED_SEND_DIMENSIONS:
            kwargs["dimensions"] = EMBED_DIM
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                response = _client().embeddings.create(**kwargs)
                break
            except Exception as e:
                if attempt == 4 or not _retryable(e):
                    raise EmbeddingError(f"{type(e).__name__}: {e}") from None
                wait = 2 ** attempt * 3
                log.warning("embedding request failed (%s); retrying in %ds", type(e).__name__, wait)
                time.sleep(wait)
        usage = extract_usage(response)
        if usage and usage["prompt"] is not None and usage["completion"] is None:
            usage["completion"] = 0
        record("embed", "embed", EMBED_MODEL, host, usage, sum(est_tokens(t) for t in batch), 0,
               round((time.perf_counter() - t0) * 1000), None)
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)


def embed(texts: list[str], kind: str = "doc") -> np.ndarray:
    prefix = EMBED_QUERY_PREFIX if kind == "query" else EMBED_DOC_PREFIX
    texts = [prefix + t for t in texts] if prefix else list(texts)
    arr = _remote(texts) if EMBED_BASE_URL else _local(texts)
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise EmbeddingError(
            f"{EMBED_MODEL} returned {arr.shape[-1]}-dimensional vectors but EMBED_DIM={EMBED_DIM}. "
            "Set EMBED_DIM to the model's size (or EMBED_SEND_DIMENSIONS=1 if it supports a dimensions parameter); "
            "an existing chunks table must be recreated when the size changes.")
    return arr
