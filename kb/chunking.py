"""Pure-Python text cleaning + heading-aware chunking (no PDF/DB imports, so it is unit-testable)."""
import re
from collections import Counter

HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{2,80})$")
DOT_LEADER = re.compile(r"\.{4,}")          # table-of-contents lines
SKIP_TITLE = re.compile(r"legal notices", re.I)


def _norm(line: str) -> str:
    return re.sub(r"^\d{1,4}|\d{1,4}$", "", line.strip()).strip()


def clean_pages(pages: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Drop blank lines, TOC dot-leader lines, bare page numbers and running headers/footers
    (any line, ignoring a leading/trailing page number, that repeats on >=40% of pages)."""
    n = len(pages)
    counts: Counter = Counter()
    for _, lines in pages:
        counts.update({_norm(l) for l in lines if l.strip()})
    thresh = max(3, int(0.4 * n))
    out = []
    for pno, lines in pages:
        keep = []
        for l in lines:
            s = l.strip()
            if not s or DOT_LEADER.search(s) or s == str(pno):
                continue
            nl = _norm(s)
            if n >= 4 and nl and len(nl) < 80 and counts[nl] >= thresh:
                continue
            keep.append((pno, s))
        out.append((pno, keep))
    return out


def split_sections(cleaned) -> list[dict]:
    """Group lines under numbered headings like '5.1.6 The customer Table'."""
    sections: list[dict] = []
    cur = {"number": "", "title": "Front matter", "lines": []}
    for _, lines in cleaned:
        for pno, s in lines:
            m = HEADING.match(s)
            if m and not s.endswith(".") and "(" not in s:
                if cur["lines"]:
                    sections.append(cur)
                cur = {"number": m.group(1), "title": m.group(2).strip(), "lines": [], "page": pno}
            else:
                if not cur["lines"] and "page" not in cur:
                    cur["page"] = pno
                cur["lines"].append((pno, s))
    if cur["lines"]:
        sections.append(cur)
    return sections


def _pack(lines, size: int, overlap_lines: int = 2):
    buf, length = [], 0
    for item in lines:
        if buf and length + len(item[1]) > size:
            yield buf
            buf = buf[-overlap_lines:]
            length = sum(len(x[1]) for x in buf)
        buf.append(item)
        length += len(item[1]) + 1
    if buf:
        yield buf


def section_chunks(pages, doc_title: str, size: int = 1800) -> list[dict]:
    """pages: [(page_no, [raw lines])]  ->  [{page, section, text}]"""
    out = []
    for sec in split_sections(clean_pages(pages)):
        if SKIP_TITLE.search(sec["title"]):
            continue
        label = f"{sec['number']} {sec['title']}".strip()
        for buf in _pack(sec["lines"], size):
            body = "\n".join(s for _, s in buf)
            if len(body) < 30:
                continue
            out.append({"page": buf[0][0], "section": label,
                        "text": f"[{doc_title} — {label}]\n{body}"})
    return out
