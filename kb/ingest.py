import hashlib, os, pathlib
from psycopg.types.json import Jsonb
from .config import IMAGE_DIR, MYSQL_DATABASE
from .db import connect
from .embed import embed
from .chunking import section_chunks
from .pdf_parser import parse_pdf
from .vision import describe_diagram, VisionError
from .mysql_schema import build_cards


def _store(conn, filename: str, sha: str, pending: list[tuple]) -> None:
    """pending items: (page, kind, content, image_path, metadata). Replaces any older version of `filename`."""
    conn.execute("DELETE FROM documents WHERE filename=%s OR sha256=%s", (filename, sha))
    doc_id = conn.execute("INSERT INTO documents(filename, sha256) VALUES (%s,%s) RETURNING id",
                          (filename, sha)).fetchone()[0]
    vecs = embed([p[2] for p in pending])
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks(document_id,page,kind,content,image_path,embedding,metadata)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(doc_id, pg, kind, content, img, v, Jsonb({"filename": filename, **meta}))
             for (pg, kind, content, img, meta), v in zip(pending, vecs)])


def ingest_pdf(path: str, vision: bool = True, force: bool = False) -> None:
    p = pathlib.Path(path)
    raw = p.read_bytes()
    sha_vision = hashlib.sha256(raw + b"v").hexdigest()
    sha_text = hashlib.sha256(raw + b"t").hexdigest()
    conn = connect()
    done = sha_vision if vision else sha_text
    if not force and conn.execute("SELECT 1 FROM documents WHERE sha256=%s", (done,)).fetchone():
        print(f"skip (already ingested): {p.name}")
        return
    title, pages, figures = parse_pdf(str(p), vision=vision)
    pending = [(c["page"], "text", c["text"], None, {"section": c["section"]})
               for c in section_chunks(pages, title)]
    page_text = {n: "\n".join(lines) for n, lines in pages}
    os.makedirs(IMAGE_DIR, exist_ok=True)
    vision_failed = False
    for i, fig in enumerate(figures, start=1):
        print(f"  p.{fig.page}: {fig.source} figure #{i} -> vision model")
        try:
            desc = describe_diagram(fig.png, page_text.get(fig.page, ""))
        except VisionError as e:
            print(f"\n  !! Vision model failed, skipping diagrams (text is still ingested):\n     {e}\n"
                  "     Fix VLM_MODEL / VLM_API_KEY in .env, check with `python -m kb status --deep`, then re-run\n"
                  "     `python -m kb ingest pdfs` (it will redo this PDF with diagrams).\n")
            vision_failed = True
            break
        if not desc:
            print("    (no diagram found / model unavailable)")
            continue
        img = f"{IMAGE_DIR}/{p.stem}_p{fig.page}_{i}.png"
        pathlib.Path(img).write_bytes(fig.png)
        cap = f" — {fig.caption}" if fig.caption else ""
        pending.append((fig.page, "diagram", f"[Diagram on page {fig.page} of {p.name}{cap}]\n{desc}",
                        img, {"caption": fig.caption}))
    if not pending:
        print(f"nothing extracted from {p.name} (scanned PDF? add OCR)")
        return
    _store(conn, p.name, sha_text if (vision_failed or not vision) else sha_vision, pending)
    n_d = sum(1 for c in pending if c[1] == "diagram")
    print(f"ingested {p.name}: {len(pending)} chunks ({n_d} diagram)")


def ingest_path(target: str, **kw) -> None:
    t = pathlib.Path(target)
    for f in (sorted(t.glob("**/*.pdf")) if t.is_dir() else [t]):
        ingest_pdf(str(f), **kw)


def sync_schema(force: bool = False) -> None:
    """Read the live MySQL schema and store one searchable card per table/view in pgvector."""
    cards = build_cards()
    name = f"mysql://{MYSQL_DATABASE}"
    sha = hashlib.sha256("\n".join(c["text"] for c in cards).encode()).hexdigest()
    conn = connect()
    if not force and conn.execute("SELECT 1 FROM documents WHERE sha256=%s", (sha,)).fetchone():
        print(f"schema cards up to date ({len(cards)} tables/views)")
        return
    pending = [(0, "schema", c["text"], None,
                {"table": c["table"], "object": c["kind"], "neighbors": c["neighbors"]}) for c in cards]
    _store(conn, name, sha, pending)
    print(f"synced {len(cards)} schema cards from {name}")
