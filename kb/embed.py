import logging
import time
from functools import lru_cache
import numpy as np
from .config import EMBED_MODEL

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    log.info("loading embedding model %s (first use downloads it)", EMBED_MODEL)
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    log.info("model ready in %.1fs", time.perf_counter() - t0)
    return model


def embed(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True, batch_size=16,
                           show_progress_bar=False).astype("float32")
