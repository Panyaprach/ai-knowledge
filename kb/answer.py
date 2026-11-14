"""RAG + text-to-SQL: retrieve docs/diagram/schema context from pgvector, then either answer from the
documentation or write a SELECT, run it on MySQL (read-only), self-correct on error, and summarise.
Every step is recorded in a decision trail (kb/trace.py)."""
import json, time
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse
from openai import OpenAI
from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, RETRIEVE_K, MAX_ROWS, EMBED_MODEL
from .retrieve import retrieve
from .sqlguard import extract_sql, UnsafeSQL
from .trace import Trace, cap, extract_reasoning, strip_think

SYSTEM = """You are a data assistant for a MySQL 8 database. Context items are labelled:
[doc FILE p.N]      documentation text (business rules, meaning of columns, design notes)
[diagram FILE p.N]  text transcribed from a schema diagram image
[schema TABLE]      exact table/column/foreign-key info read from the live database

Decide what the user needs:
1. A question about the DATA (counts, lists, totals, rankings, "which customers/films ...") -> reply with ONE ```sql block
   containing a single read-only SELECT, and nothing else. Use ONLY table and column names that appear in [schema ...]
   items. Follow business rules from the docs (e.g. use active = TRUE for current customers when relevant).
   Join through bridge tables shown in the foreign keys. Add LIMIT 50 to listings. Use MySQL syntax.
2. Anything else (concepts, design, relationships, how-to, stored procedures) -> answer in plain text from the context,
   cite sources like [sakila-en_a4.pdf p.7], and do NOT write SQL. If the context is insufficient, say so."""

SUMMARY = ("Answer the user's question in 1-4 sentences using ONLY the SQL result. Mention the key numbers or names. "
           "If the result is empty, say so. If it was truncated, say it is a partial list.")


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=300)


def _usage(r) -> dict | None:
    u = getattr(r, "usage", None)
    if not u:
        return None
    d = {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None)}
    details = getattr(u, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details else None
    if rt is not None:
        d["reasoning"] = rt
    return d


def _chat(messages, max_tokens: int = 900, tr: Trace | None = None, purpose: str = "") -> str:
    t0 = time.perf_counter()
    r = _client().chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=max_tokens, messages=messages)
    choice = r.choices[0]
    raw = choice.message.content or ""
    thinking, source = extract_reasoning(choice.message)   # whatever the provider returns; often nothing
    if tr is not None:
        tr.event("llm_call", purpose=purpose, model=LLM_MODEL,
                 messages=[{"role": m["role"], "content": cap(m["content"])} for m in messages],
                 reply_raw=cap(raw), reasoning=cap(thinking), reasoning_source=source,
                 finish_reason=choice.finish_reason, usage=_usage(r), ms=round((time.perf_counter() - t0) * 1000))
    return strip_think(raw)


def _label(h: dict) -> str:
    if h["kind"] == "schema":
        return f"schema {h['metadata'].get('table', '?')}"
    return f"{'diagram' if h['kind'] == 'diagram' else 'doc'} {h['filename']} p.{h['page']}"


def build_context(hits: list[dict], per_chunk: int = 1800) -> str:
    return "\n\n".join(f"[{_label(h)}]\n{h['content'][:per_chunk]}" for h in hits)


@dataclass
class Answer:
    question: str
    text: str = ""
    sql: str | None = None
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    hits: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    trace_id: int | None = None


def _hit_summary(h: dict) -> dict:
    return {"id": h["id"], "kind": h["kind"], "source": h["filename"], "page": h["page"],
            "table": h["metadata"].get("table"), "score": round(float(h["score"]), 4),
            "expanded": bool(h.get("expanded")), "named": bool(h.get("named")), "preview": h["content"][:160].replace("\n", " ")}


def _run(a: Answer, tr: Trace, k: int, execute: bool) -> str:
    """The pipeline. Returns the outcome label; fills `a` and the trace as it goes."""
    from .mysql_schema import run_select

    t0 = time.perf_counter()
    a.hits = retrieve(a.question, k)
    tr.event("retrieve", question=a.question, k=k, ms=round((time.perf_counter() - t0) * 1000),
             hits=[_hit_summary(h) for h in a.hits])
    a.steps.append(f"retrieved {len(a.hits)} chunks: " + ", ".join(sorted({h['kind'] for h in a.hits})))

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{build_context(a.hits)}\n\nQuestion: {a.question}"}]
    reply = _chat(msgs, tr=tr, purpose="route: answer from docs or write SQL")
    a.sql = extract_sql(reply)
    if not a.sql:
        tr.event("decision", route="docs", why="the reply had no ```sql block, so it is treated as a documentation answer")
        a.text = reply
        a.steps.append("answered from documentation (no SQL needed)")
        return "answered_from_docs"
    tr.event("decision", route="sql", why="the reply contained a SQL query")
    a.steps.append("model wrote SQL")
    if not execute:
        a.text = "(SQL generated; execution disabled)"
        return "sql_not_executed"

    for attempt in range(1, 4):
        t1 = time.perf_counter()
        sql_in = a.sql
        try:
            a.columns, a.rows, a.truncated, a.sql = run_select(a.sql, MAX_ROWS)  # a.sql = guarded SQL that actually ran
            a.error = None
            tr.event("sql_attempt", attempt=attempt, sql_in=sql_in, outcome="ok", executed_sql=a.sql, error=None,
                     n_rows=len(a.rows), truncated=a.truncated, columns=a.columns, rows_preview=a.rows[:5],
                     ms=round((time.perf_counter() - t1) * 1000))
            break
        except Exception as e:  # UnsafeSQL or MySQL error -> let the model repair it
            a.error = f"{type(e).__name__}: {e}"
            kind = "rejected by the SQL guard" if isinstance(e, UnsafeSQL) else "MySQL error"
            tr.event("sql_attempt", attempt=attempt, sql_in=sql_in, outcome=kind, executed_sql=None, error=a.error,
                     ms=round((time.perf_counter() - t1) * 1000))
            a.steps.append(f"attempt {attempt} failed ({kind}): {a.error}")
            if attempt == 3:
                break
            fix = _chat(msgs + [{"role": "assistant", "content": f"```sql\n{a.sql}\n```"},
                                {"role": "user", "content": f"That query failed: {a.error}\n"
                                 "Fix it. Reply with ONE ```sql block only."}],
                        tr=tr, purpose=f"repair after attempt {attempt}")
            new_sql = extract_sql(fix)
            if not new_sql:
                tr.event("decision", route="give_up", why="the repair reply had no ```sql block")
                break
            a.sql = new_sql
    if a.error:
        a.text = f"I couldn't get a valid query to run. Last error: {a.error}"
        return "sql_failed"
    a.steps.append(f"query returned {len(a.rows)} rows" + (" (truncated)" if a.truncated else ""))
    preview = json.dumps({"columns": a.columns, "rows": a.rows[:20], "truncated": a.truncated,
                          "total_rows_shown": len(a.rows)}, default=str)
    a.text = _chat([{"role": "system", "content": SUMMARY},
                    {"role": "user", "content": f"Question: {a.question}\nSQL:\n{a.sql}\nResult: {preview}"}],
                   400, tr=tr, purpose="summarise the SQL result")
    return "answered_with_sql"


def ask(question: str, k: int = RETRIEVE_K, execute: bool = True) -> Answer:
    a = Answer(question)
    tr = Trace(question, {"llm_model": LLM_MODEL, "llm_host": urlparse(LLM_BASE_URL).netloc,
                          "embed_model": EMBED_MODEL, "k": k, "execute": execute})
    try:
        outcome = _run(a, tr, k, execute)
    except Exception as e:  # keep the evidence of what happened, then let the caller see the error
        tr.event("error", error=f"{type(e).__name__}: {e}")
        a.trace_id = tr.save("error", sql=a.sql, error=f"{type(e).__name__}: {e}")
        raise
    a.trace_id = tr.save(outcome, answer=a.text, sql=a.sql, error=a.error)
    return a
