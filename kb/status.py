from openai import OpenAI
from . import config as C


def _endpoint(name, base, key, model):
    try:
        ids = [m.id for m in OpenAI(base_url=base, api_key=key, timeout=5).models.list().data]
    except Exception as e:
        return False, f"{name}: UNREACHABLE at {base} ({type(e).__name__})"
    note = "" if model in ids else f"  (model '{model}' not listed - e.g. `ollama pull {model}`)"
    return True, f"{name}: ok at {base}, model {model}{note}"


def collect() -> list[tuple[bool, str]]:
    out = []
    try:
        from .db import chunk_counts
        out.append((True, f"pgvector: ok, chunks by kind = {chunk_counts() or 'none yet'}"))
    except Exception as e:
        out.append((False, f"pgvector: FAIL ({type(e).__name__}: {e})"))
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
    out.append(_endpoint("vision model", C.VLM_BASE_URL, C.VLM_API_KEY, C.VLM_MODEL))
    out.append(_endpoint("answer LLM", C.LLM_BASE_URL, C.LLM_API_KEY, C.LLM_MODEL))
    return out


def run() -> int:
    res = collect()
    for ok, line in res:
        print(("[ok]   " if ok else "[FAIL] ") + line)
    return 0 if all(ok for ok, _ in res) else 1
