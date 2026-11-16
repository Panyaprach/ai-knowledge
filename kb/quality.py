"""Sanity checks on model output (pure Python)."""


def looks_degenerate(text: str, min_lines: int = 12, max_unique_line_ratio: float = 0.5,
                     min_words: int = 60, max_unique_gram_ratio: float = 0.3) -> bool:
    """True when the text is a few lines or word sequences repeated over and over."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= min_lines and len(set(lines)) / len(lines) < max_unique_line_ratio:
        return True
    words = text.split()
    if len(words) >= min_words:
        grams = [tuple(words[i:i + 6]) for i in range(len(words) - 5)]
        if len(set(grams)) / len(grams) < max_unique_gram_ratio:
            return True
    return False
