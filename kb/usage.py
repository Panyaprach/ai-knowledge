"""Records token usage of every model call in the llm_usage table."""
import logging
from collections import defaultdict

from .config import EMBED_BASE_URL, USAGE_ENABLED

USAGE_DDL = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind              TEXT NOT NULL,
    purpose           TEXT NOT NULL,
    model             TEXT NOT NULL,
    host              TEXT,
    prompt_tokens     INT,
    completion_tokens INT,
    reasoning_tokens  INT,
    est_prompt_tokens INT,
    estimated         BOOLEAN NOT NULL DEFAULT FALSE,
    finish_reason     TEXT,
    ms                INT
);
CREATE INDEX IF NOT EXISTS llm_usage_ts ON llm_usage (ts DESC);
"""

log = logging.getLogger(__name__)
CONTEXT_CAPS = (2048, 4096, 8192, 16384, 32768)
RUN = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
_ensured = False
_warned = False


def est_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def extract_usage(response) -> dict | None:
    u = getattr(response, "usage", None)
    if not u:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None),
            "reasoning": getattr(details, "reasoning_tokens", None) if details else None}


def normalize(usage: dict | None, est_prompt: int, est_completion: int) -> dict:
    if usage and usage.get("prompt") is not None:
        completion = usage.get("completion")
        return {"prompt": usage["prompt"], "completion": est_completion if completion is None else completion,
                "reasoning": usage.get("reasoning"), "estimated": completion is None}
    return {"prompt": est_prompt, "completion": est_completion, "reasoning": None, "estimated": True}


def sum_events(events: list[dict]) -> dict:
    t = {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0}
    for e in events:
        u = e.get("usage") if e.get("stage") == "llm_call" else None
        if u:
            t["calls"] += 1
            t["prompt"] += u.get("prompt") or 0
            t["completion"] += u.get("completion") or 0
            t["reasoning"] += u.get("reasoning") or 0
    return t


def flags(prompt_tokens, est_prompt_tokens, finish_reason, kind) -> list[str]:
    """Hints that something is silently going wrong. They are hints, not proof."""
    out = []
    if finish_reason == "length":
        out.append("reply was cut off by max_tokens")
    if prompt_tokens and kind != "embed":
        for cap in CONTEXT_CAPS:
            if cap - 64 <= prompt_tokens <= cap:
                out.append(f"prompt tokens are right at {cap}: the context window may be truncating the prompt")
        if kind == "llm" and est_prompt_tokens and prompt_tokens < 0.6 * est_prompt_tokens:
            out.append("provider counted far fewer prompt tokens than the prompt contains: possible truncation")
    return out


def run_summary() -> str:
    if not RUN:
        return "no model calls in this run"
    parts = [f"{k}: {v['calls']} call(s), {v['prompt']:,} in / {v['completion']:,} out" for k, v in RUN.items()]
    return "tokens this run -> " + "; ".join(parts) + "; embeddings: " + ("counted above" if EMBED_BASE_URL else "local, 0 API tokens")


def format_summary(rows: list[dict], flagged: list[dict], days: int) -> str:
    if not rows:
        return f"No model calls recorded in the last {days} days."
    head = f"{'day':<11}{'kind':<8}{'purpose':<10}{'model':<26}{'calls':>6}{'prompt':>10}{'completion':>12}{'reasoning':>10}{'avg s':>7}"
    out = [f"Token usage, last {days} days (a trailing ~ means some counts are estimated)", head, "-" * len(head)]
    for r in rows:
        out.append(f"{str(r['day']):<11}{r['kind']:<8}{r['purpose']:<10}{r['model'][:25]:<26}{r['calls']:>6}"
                   f"{r['prompt']:>10,}{r['completion']:>12,}{r['reasoning']:>10,}{r['avg_ms'] / 1000:>7.1f}"
                   + (" ~" if r["n_estimated"] else ""))
    out += ["", f"Total: {sum(r['calls'] for r in rows)} calls, {sum(r['prompt'] for r in rows):,} prompt + "
                f"{sum(r['completion'] for r in rows):,} completion tokens"]
    if flagged:
        out += ["", f"Warnings ({len(flagged)} recent call(s)); most recent first:"]
        for f in flagged[:8]:
            out.append(f"  {str(f['ts'])[:19]}  {f['kind']}/{f['purpose']}  prompt={f['prompt_tokens']}  -> " + "; ".join(f["flags"]))
    return "\n".join(out)


def record(kind: str, purpose: str, model: str, host: str, usage: dict | None, est_prompt: int,
           est_completion: int, ms: int, finish_reason: str | None) -> dict:
    """Store one model call. Never raises: logging must not break the app. Returns the normalised usage dict."""
    global _ensured, _warned
    n = normalize(usage, est_prompt, est_completion)
    RUN[kind]["calls"] += 1
    RUN[kind]["prompt"] += n["prompt"] or 0
    RUN[kind]["completion"] += n["completion"] or 0
    if not USAGE_ENABLED:
        return n
    try:
        from .db import connect
        with connect() as conn:
            if not _ensured:
                conn.execute(USAGE_DDL)
                _ensured = True
            conn.execute(
                "INSERT INTO llm_usage(kind,purpose,model,host,prompt_tokens,completion_tokens,reasoning_tokens,"
                "est_prompt_tokens,estimated,finish_reason,ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (kind, purpose, model, host, n["prompt"], n["completion"], n["reasoning"], est_prompt,
                 n["estimated"], finish_reason, ms))
    except Exception as e:
        if not _warned:
            log.warning("could not save token usage: %s: %s", type(e).__name__, e)
            _warned = True
    return n


def usage_rows(days: int = 7) -> list[dict]:
    from .db import connect
    cols = ["day", "kind", "purpose", "model", "calls", "prompt", "completion", "reasoning", "avg_ms", "n_estimated"]
    with connect() as conn:
        conn.execute(USAGE_DDL)
        rows = conn.execute(
            "SELECT ts::date, kind, purpose, model, count(*), coalesce(sum(prompt_tokens),0), "
            "coalesce(sum(completion_tokens),0), coalesce(sum(reasoning_tokens),0), coalesce(avg(ms),0)::int, "
            "count(*) FILTER (WHERE estimated) FROM llm_usage WHERE ts >= now() - make_interval(days => %s) "
            "GROUP BY 1,2,3,4 ORDER BY 1 DESC, 2, 3", (days,)).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def recent_flags(limit: int = 300) -> list[dict]:
    from .db import connect
    with connect() as conn:
        conn.execute(USAGE_DDL)
        rows = conn.execute("SELECT ts, kind, purpose, prompt_tokens, est_prompt_tokens, finish_reason FROM llm_usage "
                            "ORDER BY id DESC LIMIT %s", (limit,)).fetchall()
    out = []
    for ts, kind, purpose, p, est, fin in rows:
        f = flags(p, est, fin, kind)
        if f:
            out.append({"ts": ts, "kind": kind, "purpose": purpose, "prompt_tokens": p, "flags": f})
    return out
