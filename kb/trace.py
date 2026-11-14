"""Decision trail: a saved, human-readable record of HOW each question was answered.

One row per question in `query_log`. Its `events` column is the ordered trail:
  retrieve     what was fetched from pgvector (kind, source, page/table, score, graph-expanded or not)
  llm_call     the exact prompt, the raw reply, the model's THINKING if the provider returned any, finish_reason, tokens
  decision     which route was taken (docs answer vs SQL) and why
  sql_attempt  each SQL try: guard rejection / MySQL error / success, executed SQL, rows preview
  error        an exception, if one stopped the run
This is deliberately not distributed tracing: no extra service, just a table you can read in Adminer or the History page.
"""
import re, time

from .config import TRACE_ENABLED, TRACE_MAX_CHARS

QUERY_LOG_DDL = """
CREATE TABLE IF NOT EXISTS query_log (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    question   TEXT NOT NULL,
    outcome    TEXT NOT NULL,      -- answered_from_docs | answered_with_sql | sql_failed | sql_not_executed | error
    total_ms   INT,
    answer     TEXT,
    final_sql  TEXT,
    error      TEXT,
    meta       JSONB NOT NULL DEFAULT '{}',   -- models, k, execute flag
    events     JSONB NOT NULL DEFAULT '[]'    -- ordered decision trail: retrieve, llm_call, decision, sql_attempt, ...
);
CREATE INDEX IF NOT EXISTS query_log_created ON query_log (created_at DESC);
"""

THINK_RE = re.compile(r"(?s)<think>(.*?)</think>")
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")  # names used by different providers
_ensured = False


def cap(value, limit: int = TRACE_MAX_CHARS):
    if value is None:
        return None
    s = str(value)
    return s if len(s) <= limit else s[:limit] + f"... [truncated {len(s) - limit} chars]"


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def extract_reasoning(message):
    """Return (thinking_text, where_it_came_from). Many models/providers return nothing: then (None, None).
    Sources: a separate message field (reasoning_content / reasoning / thinking) and/or <think>...</think> in the content."""
    parts, sources = [], []
    for name in REASONING_FIELDS:
        value = getattr(message, name, None)
        if not value and getattr(message, "model_extra", None):
            value = message.model_extra.get(name)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            sources.append(f"message.{name}")
            break
    found = THINK_RE.findall(getattr(message, "content", None) or "")
    if found:
        parts.append("\n\n".join(x.strip() for x in found if x.strip()))
        sources.append("<think> tags in content")
    parts = [p for p in parts if p]
    return ("\n\n---\n\n".join(parts) or None), (", ".join(sources) or None)


class Trace:
    def __init__(self, question: str, meta: dict | None = None):
        self.question, self.meta, self.events = question, meta or {}, []
        self._t0 = time.perf_counter()

    def event(self, stage: str, **data) -> None:
        self.events.append({"t_ms": round((time.perf_counter() - self._t0) * 1000), "stage": stage, **data})

    def total_ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)

    def save(self, outcome: str, answer=None, sql=None, error=None):
        """Persist the trail. Logging must never break the app, so any failure is printed and swallowed."""
        if not TRACE_ENABLED:
            return None
        global _ensured
        try:
            from psycopg.types.json import Jsonb
            from .db import connect
            with connect() as conn:
                if not _ensured:
                    conn.execute(QUERY_LOG_DDL)
                    _ensured = True
                return conn.execute(
                    "INSERT INTO query_log(question,outcome,total_ms,answer,final_sql,error,meta,events)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                    (self.question, outcome, self.total_ms(), cap(answer), cap(sql), cap(error),
                     Jsonb(self.meta), Jsonb(self.events))).fetchone()[0]
        except Exception as e:
            print(f"[trace] could not save decision trail: {type(e).__name__}: {e}")
            return None


# ---------- reading ----------
def list_traces(n: int = 30) -> list[dict]:
    from .db import connect
    cols = ["id", "created_at", "question", "outcome", "total_ms"]
    with connect() as conn:
        conn.execute(QUERY_LOG_DDL)
        rows = conn.execute("SELECT id,created_at,question,outcome,total_ms FROM query_log "
                            "ORDER BY id DESC LIMIT %s", (n,)).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def get_trace(trace_id: int) -> dict | None:
    from .db import connect
    cols = ["id", "created_at", "question", "outcome", "total_ms", "answer", "final_sql", "error", "meta", "events"]
    with connect() as conn:
        row = conn.execute("SELECT id,created_at,question,outcome,total_ms,answer,final_sql,error,meta,events "
                           "FROM query_log WHERE id=%s", (trace_id,)).fetchone()
    return dict(zip(cols, row)) if row else None


def format_trace(t: dict, full: bool = False) -> str:
    """Plain-text rendering for the terminal (`python -m kb trace ID`)."""
    short = (lambda s, n=500: s if full or len(s) <= n else s[:n] + f" ...[{len(s) - n} more chars; use --full]")
    out = [f"Trace #{t['id']}  {str(t['created_at'])[:19]}  outcome={t['outcome']}  total={t['total_ms']} ms",
           f"Question : {t['question']}", f"Meta     : {t['meta']}", ""]
    for ev in t["events"]:
        stage, at = ev["stage"], f"[{ev['t_ms']:>6} ms]"
        if stage == "retrieve":
            out.append(f"{at} RETRIEVE  {len(ev['hits'])} chunks in {ev.get('ms', '?')} ms")
            for h in ev["hits"]:
                where = f"table {h['table']}" if h["kind"] == "schema" else f"{h['source']} p.{h['page']}"
                out.append(f"            - {h['kind']:<7} {where:<34} score={h['score']}"
                           f"{'  (added via foreign key)' if h['expanded'] else ''}")
        elif stage == "llm_call":
            u = ev.get("usage") or {}
            out.append(f"{at} LLM CALL  purpose={ev['purpose']}  model={ev['model']}  {ev['ms']} ms  "
                       f"finish={ev['finish_reason']}  tokens={u or 'n/a'}")
            for m in ev["messages"]:
                out.append(f"            prompt[{m['role']}]: " + short(m["content"]).replace("\n", "\n              "))
            if ev.get("reasoning"):
                out.append(f"            THINKING (from {ev['reasoning_source']}):\n              "
                           + cap(ev["reasoning"], 4000 if not full else 10**9).replace("\n", "\n              "))
            else:
                out.append("            THINKING: none returned by this model/provider")
            out.append("            reply: " + short(ev["reply_raw"] or "", 1500).replace("\n", "\n              "))
            if ev["finish_reason"] == "length":
                out.append("            !! finish_reason=length: the reply was cut off by max_tokens "
                           "(thinking models can spend the whole budget on thinking)")
        elif stage == "decision":
            out.append(f"{at} DECISION  route={ev['route']}  because: {ev['why']}")
        elif stage == "sql_attempt":
            out.append(f"{at} SQL TRY #{ev['attempt']}  -> {ev['outcome']}  ({ev['ms']} ms)")
            out.append("            sql in : " + ev["sql_in"].replace("\n", "\n                     "))
            if ev.get("error"):
                out.append(f"            error  : {ev['error']}")
            if ev.get("executed_sql"):
                out.append("            ran    : " + ev["executed_sql"].replace("\n", "\n                     "))
                out.append(f"            rows   : {ev['n_rows']}  columns={ev['columns']}  preview={ev['rows_preview']}")
        elif stage == "error":
            out.append(f"{at} ERROR     {ev['error']}")
        else:
            out.append(f"{at} {stage.upper()}  { {k: v for k, v in ev.items() if k not in ('t_ms', 'stage')} }")
    out += ["", "Final SQL: " + (t["final_sql"] or "-"), "Answer   : " + (t["answer"] or "-")]
    if t["error"]:
        out.append("Error    : " + t["error"])
    return "\n".join(out)
