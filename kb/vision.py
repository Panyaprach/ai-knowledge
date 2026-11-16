"""Turn a figure into searchable text using any OpenAI-compatible vision model."""
import base64, logging, time
from functools import lru_cache
from openai import (OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
                    BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
from .config import (VLM_BASE_URL, VLM_API_KEY, VLM_MODEL, VLM_DELAY_SEC, VLM_TEMPERATURE, VLM_MAX_TOKENS,
                     VLM_FREQUENCY_PENALTY)
from urllib.parse import urlparse
from .quality import looks_degenerate
from .usage import record, extract_usage, est_tokens

PROMPT = """You are reading a figure from database documentation. It may be an ER / EER diagram or a schema overview
(boxes = tables/entities, lines = relationships, often labelled like fk_city_country).
Transcribe ONLY what is visible:

## Tables / entities
Every table or entity box you can read, one per line (add attributes only if they are shown).
## Relationships
One line per connecting line: A --label--> B. Give cardinality (1, 0..1, 1..*, many) ONLY if the line-end symbols are clearly legible; otherwise write "cardinality unclear".
## Views
Any boxes that look like database views, if present.
## Mermaid
A ```mermaid erDiagram``` block with the entities and relationships above.

Never invent names. List each table and each relationship once; do not repeat lines.{hint_block}
If the image contains NO diagram (a logo, chart, or plain text), reply with exactly: NO_DIAGRAM"""


log = logging.getLogger(__name__)


class VisionError(RuntimeError):
    """The vision endpoint rejected the request, or its output was unusable. Retrying the same call will not help."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=300)


def describe_raw(png: bytes, hint: str = "", temperature: float | None = None):
    """One model call. Returns (text, info) where info has finish_reason, token counts and time."""
    b64 = base64.b64encode(png).decode()
    hint_block = f"\n\nText extracted from the page (use it to spell names correctly):\n---\n{hint[:1000]}\n---" if hint.strip() else ""
    kwargs = dict(
        model=VLM_MODEL, max_tokens=VLM_MAX_TOKENS,
        temperature=VLM_TEMPERATURE if temperature is None else temperature,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(hint_block=hint_block)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
    if VLM_FREQUENCY_PENALTY:
        kwargs["frequency_penalty"] = VLM_FREQUENCY_PENALTY
    for attempt in range(5):
        t0 = time.perf_counter()
        try:
            r = _client().chat.completions.create(**kwargs)
            time.sleep(VLM_DELAY_SEC)
            choice = r.choices[0]
            out = (choice.message.content or "").strip()
            ms = round((time.perf_counter() - t0) * 1000)
            n = record("vision", "diagram", VLM_MODEL, urlparse(VLM_BASE_URL).netloc, extract_usage(r),
                       est_tokens(PROMPT), est_tokens(out), ms, choice.finish_reason)
            return out, {"finish_reason": choice.finish_reason, "temperature": kwargs["temperature"],
                         "prompt_tokens": n["prompt"], "completion_tokens": n["completion"],
                         "estimated": n["estimated"], "ms": ms}
        except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            raise VisionError(f"{type(e).__name__}: {e}") from None
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            wait = 2 ** attempt * 3
            log.warning("temporary error (%s); retrying in %ds", type(e).__name__, wait)
            time.sleep(wait)
    raise VisionError("vision model kept failing after 5 attempts")


def describe_diagram(png: bytes, hint: str = ""):
    """Returns (description or None, info). Looping output raises VisionError instead of being stored."""
    info = {}
    for temperature in (None, 0.3):
        out, info = describe_raw(png, hint, temperature)
        if not out or out.startswith("NO_DIAGRAM"):
            return None, info
        if not looks_degenerate(out):
            return out, info
        if temperature is None:
            log.warning("description looks degenerate (same lines repeated); retrying once with temperature 0.3")
    raise VisionError(
        "the model's description was degenerate (the same few lines repeated). Likely causes: the image or prompt was "
        "cut off by a small context window (Ollama defaults to 4096 tokens, see README 'Ollama context'), or the vision "
        f"model is too small to read this figure. Last call: {info}")
