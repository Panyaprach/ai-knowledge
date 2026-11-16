import re
from collections import Counter
from .db import connect
from .embed import embed
from .linking import find_named_tables

COLS = ["id", "filename", "page", "kind", "content", "image_path", "metadata", "score"]

HYBRID_SQL = """
WITH sem AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rnk
  FROM chunks ORDER BY embedding <=> %(q)s LIMIT 30
), kw AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, to_tsquery('simple', %(t)s)) DESC) AS rnk
  FROM chunks WHERE %(t)s <> '' AND tsv @@ to_tsquery('simple', %(t)s)
  ORDER BY ts_rank_cd(tsv, to_tsquery('simple', %(t)s)) DESC LIMIT 30
)
SELECT c.id, d.filename, c.page, c.kind, c.content, c.image_path, c.metadata,
       COALESCE(1.0/(60+sem.rnk),0) + COALESCE(1.0/(60+kw.rnk),0) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
LEFT JOIN sem ON sem.id = c.id
LEFT JOIN kw  ON kw.id  = c.id
WHERE sem.id IS NOT NULL OR kw.id IS NOT NULL
ORDER BY score DESC LIMIT %(k)s
"""

NEIGHBOR_SQL = """
SELECT c.id, d.filename, c.page, c.kind, c.content, c.image_path, c.metadata, 0.0
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.kind = 'schema' AND c.metadata->>'table' = ANY(%s)
"""


def search(question: str, k: int = 8) -> list[dict]:
    """Hybrid search: pgvector cosine + Postgres full-text, merged with Reciprocal Rank Fusion."""
    q = embed([question], "query")[0]
    tokens = [t for t in re.findall(r"\w+", question) if len(t) > 2]
    with connect() as conn:
        rows = conn.execute(HYBRID_SQL, {"q": q, "t": " | ".join(tokens), "k": k}).fetchall()
    return [dict(zip(COLS, r)) for r in rows]


def schema_neighbors(hits: list[dict], limit: int = 6) -> list[dict]:
    """Add schema cards of tables joined by foreign key to the tables already found."""
    found = {h["metadata"].get("table") for h in hits if h["kind"] == "schema"} - {None}
    if not found:
        return []
    votes = Counter(n for h in hits if h["kind"] == "schema"
                    for n in h["metadata"].get("neighbors", []) if n not in found)
    want = [t for t, _ in votes.most_common(limit)]
    if not want:
        return []
    with connect() as conn:
        rows = conn.execute(NEIGHBOR_SQL, (want,)).fetchall()
    return [{**dict(zip(COLS, r)), "expanded": True} for r in rows]


def named_table_cards(question: str, hits: list[dict], limit: int = 5) -> list[dict]:
    """Always include the schema card of a table the question names."""
    with connect() as conn:
        names = [r[0] for r in conn.execute(
            "SELECT DISTINCT metadata->>'table' FROM chunks WHERE kind='schema'").fetchall() if r[0]]
        have = {h["metadata"].get("table") for h in hits if h["kind"] == "schema"}
        want = [t for t in find_named_tables(question, names, limit) if t not in have]
        rows = conn.execute(NEIGHBOR_SQL, (want,)).fetchall() if want else []
    return [{**dict(zip(COLS, r)), "named": True} for r in rows]


def retrieve(question: str, k: int = 8, expand: bool = True) -> list[dict]:
    hits = search(question, k)
    if not expand:
        return hits
    hits = hits + named_table_cards(question, hits)
    return hits + schema_neighbors(hits)
