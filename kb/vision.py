"""Turn a figure into searchable text using any OpenAI-compatible vision model."""
import base64, time
from functools import lru_cache
from openai import (OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
                    BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
from .config import (VLM_BASE_URL, VLM_API_KEY, VLM_MODEL, VLM_DELAY_SEC, VLM_TEMPERATURE, VLM_MAX_TOKENS,
                     VLM_FREQUENCY_PENALTY)
from .quality import looks_degenerate

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
    for attempt in range(5):  # free tiers rate-limit; back off and retry
        t0 = time.perf_counter()
        try:
            r = _client().chat.completions.create(**kwargs)
            time.sleep(VLM_DELAY_SEC)
            choice, usage = r.choices[0], getattr(r, "usage", None)
            return (choice.message.content or "").strip(), {
                "finish_reason": choice.finish_reason, "temperature": kwargs["temperature"],
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "ms": round((time.perf_counter() - t0) * 1000)}
        except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            raise VisionError(f"{type(e).__name__}: {e}") from None
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            wait = 2 ** attempt * 3
            print(f"    VLM temporary error ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
    raise VisionError("vision model kept failing after 5 attempts")


def describe_diagram(png: bytes, hint: str = "") -> str | None:
    """Description text, None if the figure is not a diagram. Rejects looping/garbled output instead of storing it."""
    for temperature in (None, 0.3):          # second try: a little randomness often breaks a repetition loop
        out, info = describe_raw(png, hint, temperature)
        if not out or out.startswith("NO_DIAGRAM"):
            return None
        if not looks_degenerate(out):
            return out
        if temperature is None:
            print("    description looks degenerate (same lines repeated); retrying once with temperature 0.3")
    raise VisionError(
        "the model's description was degenerate (the same few lines repeated). Likely causes: the image or prompt was "
        "cut off by a small context window (Ollama defaults to 4096 tokens, see README 'Ollama context'), or the vision "
        f"model is too small to read this figure. Last call: {info}")
