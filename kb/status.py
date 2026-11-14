import base64, io
from openai import OpenAI
from . import config as C


def _hf_hint(name, base, model):
    if "huggingface.co" in base and ":" in model:
        suffix = model.split(":", 1)[1]
        if suffix not in ("fastest", "cheapest", "preferred"):
            return (False, f"{name}: '{model}' looks like an Ollama tag. On the Hugging Face router, the text after ':' "
                           f"means PROVIDER (e.g. ':hyperbolic') or ':fastest'. Use a Hub id like 'Org/Model-Name'.")
    return None


def _probe(name, base, key, model, image=False):
    """One tiny real request, so the provider's actual error message is shown."""
    try:
        content = "Reply with the single word OK."
        if image:
            from PIL import Image
            buf = io.BytesIO(); Image.new("RGB", (64, 64), "white").save(buf, "PNG")
            content = [{"type": "text", "text": "Reply with the single word OK."},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]
        OpenAI(base_url=base, api_key=key, timeout=60).chat.completions.create(
            model=model, max_tokens=5, messages=[{"role": "user", "content": content}])
        return True, f"{name} probe: request accepted by {model}"
    except Exception as e:
        return False, f"{name} probe FAILED ({type(e).__name__}): {str(e)[:400]}"


def _endpoint(name, base, key, model):
    try:
        ids = [m.id for m in OpenAI(base_url=base, api_key=key, timeout=5).models.list().data]
    except Exception as e:
        return False, f"{name}: UNREACHABLE at {base} ({type(e).__name__})"
    note = "" if model in ids else f"  (model '{model}' not listed - e.g. `ollama pull {model}`)"
    return True, f"{name}: ok at {base}, model {model}{note}"


def collect(deep: bool = False) -> list[tuple[bool, str]]:
    out = []
    try:
        from .db import chunk_counts
        out.append((True, f"pgvector: ok, chunks by kind = {chunk_counts() or 'none yet'}"))
    except Exception as e:
        hint = "  -> not initialised yet: run `python -m kb init` (or `setup`)" if "vector type not found" in str(e) else ""
        out.append((False, f"pgvector: FAIL ({type(e).__name__}: {e}){hint}"))
    try:
        from .mysql_schema import mysql_conn
        c = mysql_conn(3)
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s", (C.MYSQL_DATABASE,))
            n = cur.fetchone()[0]
        c.close()
        out.append((n > 0, f"mysql: ok, {n} tables/views in '{C.MYSQL_DATABASE}'" if n else "mysql: connected but database is empty"))
    except Exception as e:
        out.append((False, f"mysql: FAIL ({type(e).__name__}: {e})"))
    for name, base, key, model, img in (("vision model", C.VLM_BASE_URL, C.VLM_API_KEY, C.VLM_MODEL, True),
                                        ("answer LLM", C.LLM_BASE_URL, C.LLM_API_KEY, C.LLM_MODEL, False)):
        bad = _hf_hint(name, base, model)
        if bad:
            out.append(bad)
            continue
        out.append(_endpoint(name, base, key, model))
        if deep:
            out.append(_probe(name, base, key, model, image=img))
    return out


def run(deep: bool = False) -> int:
    res = collect(deep)
    for ok, line in res:
        print(("[ok]   " if ok else "[FAIL] ") + line)
    return 0 if all(ok for ok, _ in res) else 1
