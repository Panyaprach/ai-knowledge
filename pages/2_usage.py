"""Token usage per day / model / purpose, with warnings. Appears in the Streamlit sidebar."""
import pandas as pd
import streamlit as st

from kb.config import setup_logging

setup_logging()  # idempotent

from kb.usage import recent_flags, usage_rows

st.set_page_config(page_title="Token usage", page_icon="🔢", layout="wide")
st.title("🔢 Token usage")
st.caption("Every model call is counted: the vision model during ingest, the text model for each question. "
           "Embeddings run locally and use no API tokens. Counts are provider-reported unless marked estimated.")

days = st.sidebar.slider("Days", 1, 30, 7)
try:
    rows = usage_rows(days)
    flagged = recent_flags()
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.stop()
if not rows:
    st.info("No model calls recorded yet.")
    st.stop()

df = pd.DataFrame(rows)
c1, c2, c3 = st.columns(3)
c1.metric("Model calls", f"{df['calls'].sum():,}")
c2.metric("Prompt tokens", f"{df['prompt'].sum():,}")
c3.metric("Completion tokens", f"{df['completion'].sum():,}")

st.subheader("Per day")
daily = df.groupby("day")[["prompt", "completion"]].sum()
st.bar_chart(daily)
st.caption("Free tiers often limit calls per day, not tokens: see the calls column below.")

st.subheader("By model and purpose")
st.dataframe(df.rename(columns={"n_estimated": "estimated calls", "avg_ms": "avg ms"}), use_container_width=True, hide_index=True)

if flagged:
    st.subheader("⚠️ Warnings")
    st.caption("Hints that the context window may be truncating prompts, or replies were cut off. Not proof.")
    st.dataframe(pd.DataFrame([{**f, "flags": "; ".join(f["flags"])} for f in flagged]), use_container_width=True, hide_index=True)
