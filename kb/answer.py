"""RAG + text-to-SQL: retrieve docs/diagram/schema context from pgvector, then either answer from the
documentation or write a SELECT, run it on MySQL (read-only), self-correct on error, and summarise."""
import json, re
from dataclasses import dataclass, field
from functools import lru_cache
from openai import OpenAI
from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, RETRIEVE_K, MAX_ROWS
from .retrieve import retrieve
from .sqlguard import extract_sql

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


def _chat(messages, max_tokens: int = 900) -> str:
    r = _client().chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=max_tokens, messages=messages)
    text = (r.choices[0].message.content or "").strip()
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()  # reasoning models


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


def ask(question: str, k: int = RETRIEVE_K, execute: bool = True) -> Answer:
    from .mysql_schema import run_select
    a = Answer(question)
    a.hits = retrieve(question, k)
    a.steps.append(f"retrieved {len(a.hits)} chunks: " +
                   ", ".join(sorted({h['kind'] for h in a.hits})))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{build_context(a.hits)}\n\nQuestion: {question}"}]
    reply = _chat(msgs)
    a.sql = extract_sql(reply)
    if not a.sql:
        a.text = reply
        a.steps.append("answered from documentation (no SQL needed)")
        return a
    a.steps.append("model wrote SQL")
    if not execute:
        a.text = "(SQL generated; execution disabled)"
        return a

    for attempt in range(3):
        try:
            a.columns, a.rows, a.truncated = run_select(a.sql, MAX_ROWS)
            a.error = None
            break
        except Exception as e:  # UnsafeSQL or MySQL error -> let the model repair it
            a.error = f"{type(e).__name__}: {e}"
            a.steps.append(f"attempt {attempt + 1} failed: {a.error}")
            if attempt == 2:
                break
            fix = _chat(msgs + [{"role": "assistant", "content": f"```sql\n{a.sql}\n```"},
                                {"role": "user", "content": f"That query failed: {a.error}\n"
                                 "Fix it. Reply with ONE ```sql block only."}])
            new_sql = extract_sql(fix)
            if not new_sql:
                break
            a.sql = new_sql
    if a.error:
        a.text = f"I couldn't get a valid query to run. Last error: {a.error}"
        return a
    a.steps.append(f"query returned {len(a.rows)} rows" + (" (truncated)" if a.truncated else ""))
    preview = json.dumps({"columns": a.columns, "rows": a.rows[:20], "truncated": a.truncated,
                          "total_rows_shown": len(a.rows)}, default=str)
    a.text = _chat([{"role": "system", "content": SUMMARY},
                    {"role": "user", "content": f"Question: {question}\nSQL:\n{a.sql}\nResult: {preview}"}], 400)
    return a
