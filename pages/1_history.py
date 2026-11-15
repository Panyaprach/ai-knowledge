"""Decision trail viewer. Appears in the Streamlit sidebar next to the main page."""
import pandas as pd
import streamlit as st

from kb.config import setup_logging

setup_logging()  # idempotent

from kb.trace import get_trace, list_traces

st.set_page_config(page_title="Decision trail", page_icon="🔎", layout="wide")
st.title("🔎 Decision trail")
st.caption("Every question is saved with what was retrieved, the exact prompts, the model's thinking (when the provider "
           "returns it), each SQL attempt and why it was accepted or rejected.")

try:
    rows = list_traces(50)
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.stop()
if not rows:
    st.info("No questions yet. Ask something on the main page first.")
    st.stop()

BADGE = {"answered_from_docs": "📄 docs", "answered_with_sql": "🗄️ SQL", "sql_failed": "❌ SQL failed",
         "sql_not_executed": "⏸️ SQL not run", "error": "💥 error"}
label = lambda r: f"#{r['id']} · {BADGE.get(r['outcome'], r['outcome'])} · {r['question'][:70]}"
chosen = st.sidebar.selectbox("Question", [r["id"] for r in rows], format_func=lambda i: label(next(r for r in rows if r["id"] == i)))
t = get_trace(chosen)

c1, c2, c3 = st.columns(3)
c1.metric("Outcome", BADGE.get(t["outcome"], t["outcome"]))
c2.metric("Total time", f"{t['total_ms']} ms")
c3.metric("Model", t["meta"].get("llm_model", "?"), t["meta"].get("llm_host", ""))
st.subheader(t["question"])

for ev in t["events"]:
    stage, at = ev["stage"], f"{ev['t_ms']} ms"
    if stage == "retrieve":
        st.markdown(f"### 1 · Retrieve  _(at {at}, took {ev.get('ms', '?')} ms)_")
        st.caption("Hybrid search in pgvector (meaning + keywords). Rows in the *added because* column were not found by the "
                   "search: the question named the table, or it is joined by foreign key to a table that was found.")
        df = pd.DataFrame(ev["hits"])
        df["where"] = df.apply(lambda h: f"table {h['table']}" if h["kind"] == "schema" else f"{h['source']} p.{h['page']}", axis=1)
        df["added because"] = df.apply(lambda h: "named in the question" if h.get("named") else ("foreign key" if h["expanded"] else ""), axis=1)
        st.dataframe(df[["kind", "where", "score", "added because", "preview"]], use_container_width=True, hide_index=True)
    elif stage == "llm_call":
        u = ev.get("usage") or {}
        st.markdown(f"### LLM call · {ev['purpose']}  _(at {at}, {ev['ms']} ms, finish={ev['finish_reason']}, tokens={u or 'n/a'})_")
        if ev["finish_reason"] == "length":
            st.warning("The reply was cut off by max_tokens. Thinking models can spend the whole budget on thinking.")
        if ev.get("reasoning"):
            with st.expander(f"🧠 Model thinking (from {ev['reasoning_source']})", expanded=True):
                st.text(ev["reasoning"])
        else:
            st.caption("🧠 No thinking returned by this model/provider.")
        with st.expander("Reply (raw)"):
            st.text(ev["reply_raw"] or "(empty)")
        with st.expander("Prompt sent to the model"):
            for m in ev["messages"]:
                st.markdown(f"**{m['role']}**")
                st.code(m["content"], language=None)
    elif stage == "decision":
        st.info(f"**Decision** (at {at}): route = `{ev['route']}`, because {ev['why']}.")
    elif stage == "sql_attempt":
        ok = ev["outcome"] == "ok"
        (st.success if ok else st.error)(f"SQL attempt #{ev['attempt']}: {ev['outcome']}  ({ev['ms']} ms)")
        st.code(ev["sql_in"], language="sql")
        if ev.get("error"):
            st.caption(f"Error given back to the model: {ev['error']}")
        if ok:
            if ev["executed_sql"].strip() != ev["sql_in"].strip():
                st.caption("The guard rewrote the query (added LIMIT / removed comments). What actually ran:")
                st.code(ev["executed_sql"], language="sql")
            st.caption(f"{ev['n_rows']} rows" + (" (truncated)" if ev.get("truncated") else ""))
            if ev["rows_preview"]:
                st.dataframe(pd.DataFrame(ev["rows_preview"], columns=ev["columns"]), hide_index=True)
    elif stage == "error":
        st.error(f"Error at {at}: {ev['error']}")

st.markdown("### Result")
if t["final_sql"]:
    st.code(t["final_sql"], language="sql")
st.write(t["answer"] or "(no answer)")
if t["error"]:
    st.error(t["error"])
