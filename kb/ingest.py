import hashlib, io, logging, os, pathlib, time
from PIL import Image
from psycopg.types.json import Jsonb
from .config import IMAGE_DIR, MYSQL_DATABASE, VLM_MODEL
from .db import connect
from .embed import describe_backend, embed
from .chunking import section_chunks
from .pdf_parser import parse_pdf
from .vision import describe_diagram, VisionError
from .mysql_schema import build_cards

log = logging.getLogger(__name__)


def _store(conn, filename: str, sha: str, pending: list[tuple]) -> None:
    """pending items: (page, kind, content, image_path, metadata). Replaces any older version of `filename`."""
    t0 = time.perf_counter()
    vecs = embed([p[2] for p in pending])
    log.info("%d chunks embedded in %.1fs (%s)", len(pending), time.perf_counter() - t0, describe_backend())
    t0 = time.perf_counter()
    conn.execute("DELETE FROM documents WHERE filename=%s OR sha256=%s", (filename, sha))
    doc_id = conn.execute("INSERT INTO documents(filename, sha256) VALUES (%s,%s) RETURNING id",
                          (filename, sha)).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks(document_id,page,kind,content,image_path,embedding,metadata)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(doc_id, pg, kind, content, img, v, Jsonb({"filename": filename, **meta}))
             for (pg, kind, content, img, meta), v in zip(pending, vecs)])
    kinds = {k: sum(1 for p in pending if p[1] == k) for k in sorted({p[1] for p in pending})}
    log.info("%s: %d chunks %s stored in %.1fs", filename, len(pending), kinds, time.perf_counter() - t0)


def ingest_pdf(path: str, vision: bool = True, force: bool = False) -> None:
    p = pathlib.Path(path)
    raw = p.read_bytes()
    sha_vision = hashlib.sha256(raw + b"v").hexdigest()
    sha_text = hashlib.sha256(raw + b"t").hexdigest()
    conn = connect()
    done = sha_vision if vision else sha_text
    if not force and conn.execute("SELECT 1 FROM documents WHERE sha256=%s", (done,)).fetchone():
        log.info("%s: already ingested, skipping (use --force to redo)", p.name)
        return
    t0 = time.perf_counter()
    title, pages, figures = parse_pdf(str(p), vision=vision)
    chunks = section_chunks(pages, title)
    log.info("%s: %d pages -> %d text chunks, %d figure(s) (%.1fs)%s", p.name, len(pages), len(chunks), len(figures),
             time.perf_counter() - t0, "" if vision else "  [vision off]")
    pending = [(c["page"], "text", c["text"], None, {"section": c["section"]}) for c in chunks]
    page_text = {n: "\n".join(lines) for n, lines in pages}
    os.makedirs(IMAGE_DIR, exist_ok=True)
    vision_failed = False
    for i, fig in enumerate(figures, start=1):
        w, h = Image.open(io.BytesIO(fig.png)).size
        log.info("p.%d %s figure %dx%dpx -> %s (can take a minute on a local model)", fig.page, fig.source, w, h, VLM_MODEL)
        hint = page_text.get(fig.page, "") if fig.source == "vector" else ""
        try:
            desc, info = describe_diagram(fig.png, hint)
        except VisionError as e:
            log.error("vision model failed, diagrams skipped (text and schema are still ingested):\n  %s\n"
                      "  fix the model/settings (python -m kb vision-test <pdf>), then: python -m kb ingest pdfs --force",
                      str(e).replace("\n", "\n  "))
            vision_failed = True
            break
        secs = info.get("ms", 0) / 1000
        if desc is None:
            log.info("p.%d the model found no diagram in this figure (%.1fs)", fig.page, secs)
            continue
        log.info("p.%d ok: %d lines, tokens %s in / %s out%s, %.1fs, finish=%s", fig.page, len(desc.splitlines()),
                 f"{info['prompt_tokens']:,}", f"{info['completion_tokens']:,}", " (est.)" if info.get("estimated") else "",
                 secs, info["finish_reason"])
        log.debug("p.%d model output: %s", fig.page, desc[:300].replace("\n", " | "))
        img = f"{IMAGE_DIR}/{p.stem}_p{fig.page}_{i}.png"
        pathlib.Path(img).write_bytes(fig.png)
        cap = f" — {fig.caption}" if fig.caption else ""
        pending.append((fig.page, "diagram", f"[Diagram on page {fig.page} of {p.name}{cap}]\n{desc}",
                        img, {"caption": fig.caption}))
    if not pending:
        log.warning("nothing extracted from %s (scanned PDF? add OCR)", p.name)
        return
    _store(conn, p.name, sha_text if (vision_failed or not vision) else sha_vision, pending)


def ingest_path(target: str, **kw) -> None:
    t = pathlib.Path(target)
    for f in (sorted(t.glob("**/*.pdf")) if t.is_dir() else [t]):
        ingest_pdf(str(f), **kw)


def sync_schema(force: bool = False) -> None:
    """Read the live MySQL schema and store one searchable card per table/view in pgvector."""
    t0 = time.perf_counter()
    cards = build_cards()
    read_s = time.perf_counter() - t0
    name = f"mysql://{MYSQL_DATABASE}"
    sha = hashlib.sha256("\n".join(c["text"] for c in cards).encode()).hexdigest()
    conn = connect()
    if not force and conn.execute("SELECT 1 FROM documents WHERE sha256=%s", (sha,)).fetchone():
        log.info("MySQL schema: %d tables/views read in %.1fs; cards already up to date", len(cards), read_s)
        return
    log.info("MySQL schema: %d tables/views read in %.1fs", len(cards), read_s)
    pending = [(0, "schema", c["text"], None,
                {"table": c["table"], "object": c["kind"], "neighbors": c["neighbors"]}) for c in cards]
    _store(conn, name, sha, pending)
