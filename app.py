"""Demo UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sakila Knowledge Base", page_icon="🎬", layout="wide")
st.title("🎬 Sakila knowledge base")
st.caption("PDF text + schema diagram + live MySQL schema → pgvector → RAG answers & read-only text-to-SQL on MySQL")

EXAMPLES = [
    "Which table links films and actors?",
    "What does the active column in customer mean?",
    "How many films are in each category?",
    "Who are the 5 customers who paid the most in total?",
    "Which films are rented out and overdue right now?",
    "What is the known design issue with stores and staff?",
]

with st.sidebar:
    st.header("System status")
    if st.button("Check connections"):
        from kb.status import collect
        for ok, line in collect():
            (st.success if ok else st.error)(line)
    execute = st.checkbox("Run generated SQL on MySQL", value=True)
    k = st.slider("Chunks to retrieve", 4, 16, 8)

st.write("**Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state["q"] = ex

q = st.text_input("Ask about the Sakila database", key="q", placeholder="e.g. Top 10 most rented films")
if st.button("Ask", type="primary") and q.strip():
    from kb.answer import ask
    with st.spinner("Retrieving context and asking the model…"):
        try:
            a = ask(q.strip(), k=k, execute=execute)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.info("Run `python -m kb status` to see which service is down.")
            st.stop()
    st.subheader("Answer")
    st.markdown(a.text)
    if a.sql:
        st.subheader("SQL")
        st.code(a.sql, language="sql")
    if a.rows:
        st.dataframe(pd.DataFrame(a.rows, columns=a.columns), use_container_width=True)
        if a.truncated:
            st.caption("Result truncated.")
    with st.expander("How it was answered"):
        st.write(" → ".join(a.steps))
    st.subheader("Sources")
    for h in a.hits:
        label = h["metadata"].get("table") if h["kind"] == "schema" else f"p.{h['page']}"
        with st.expander(f"{h['kind']} · {h['filename']} · {label}" + (" · (graph-expanded)" if h.get("expanded") else "")):
            if h["image_path"]:
                st.image(h["image_path"], caption="Original figure from the PDF")
            st.text(h["content"])
