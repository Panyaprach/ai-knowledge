"""Spot database table names inside a question (pure Python, no DB needed)."""
import re


def find_named_tables(question: str, table_names, limit: int = 5) -> list[str]:
    """Tables named in the question (plurals tolerated), in order of appearance."""
    q = question.lower()
    hits = []
    for t in table_names:
        m = re.search(rf"(?<![a-z0-9_]){re.escape(t.lower())}(?:s|es)?(?![a-z0-9_])", q)
        if m:
            hits.append((m.start(), t))
    return [t for _, t in sorted(hits)][:limit]
