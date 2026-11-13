"""Turn a figure into searchable text using any OpenAI-compatible vision model."""
import base64, time
from functools import lru_cache
from openai import OpenAI, RateLimitError, APIError
from .config import VLM_BASE_URL, VLM_API_KEY, VLM_MODEL, VLM_DELAY_SEC

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

Never invent names. Text extracted from the page (use it to spell names correctly):
---
{hint}
---
If the image contains NO diagram (a logo, chart, or plain text), reply with exactly: NO_DIAGRAM"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY, timeout=300)


def describe_diagram(png: bytes, page_text: str) -> str | None:
    b64 = base64.b64encode(png).decode()
    content = [
        {"type": "text", "text": PROMPT.format(hint=page_text[:1500] or "(none)")},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    for attempt in range(5):  # free tiers rate-limit; back off and retry
        try:
            r = _client().chat.completions.create(
                model=VLM_MODEL, temperature=0, max_tokens=2000,
                messages=[{"role": "user", "content": content}])
            time.sleep(VLM_DELAY_SEC)
            out = (r.choices[0].message.content or "").strip()
            return None if not out or out.startswith("NO_DIAGRAM") else out
        except (RateLimitError, APIError) as e:
            wait = 2 ** attempt * 3
            print(f"    VLM error ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
    return None
