"""Retrieval-only evaluation (no LLM needed): does the top-k context contain the expected facts?"""
import json, pathlib
from .db import chunk_counts
from .retrieve import retrieve


def run(path: str = "eval/questions.json", k: int = 8) -> int:
    have = chunk_counts()
    items = json.loads(pathlib.Path(path).read_text())
    ok = ran = 0
    for it in items:
        need = it.get("needs")
        if need and not have.get(need):
            print(f"SKIP  {it['q']}   (no '{need}' chunks stored)")
            continue
        norm = lambda s: " ".join(s.lower().split())          # PDF text wraps lines; compare without line breaks
        text = norm("\n".join(h["content"] for h in retrieve(it["q"], k, expand=True)))
        miss = [e for e in it["expect"] if norm(e) not in text]
        ran += 1
        ok += not miss
        print(f"{'PASS' if not miss else 'FAIL'}  {it['q']}" + (f"   missing: {miss}" if miss else ""))
    print(f"\nretrieval hit-rate @k={k}: {ok}/{ran}")
    return 0 if ok == ran else 1
