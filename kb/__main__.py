import argparse, logging, pathlib, sys, time

log = logging.getLogger("kb.setup")


def _print_answer(a):
    print(a.text)
    if a.sql:
        print("\nSQL:\n" + a.sql)
    if a.columns:
        print("\n" + " | ".join(a.columns))
        for r in a.rows[:20]:
            print(" | ".join(str(v) for v in r))
        if len(a.rows) > 20 or a.truncated:
            print(f"... ({len(a.rows)} rows kept)")
    print("\nSteps: " + " -> ".join(a.steps))
    if a.usage and a.usage.get("calls"):
        u = a.usage
        print(f"Tokens: {u['prompt']:,} prompt + {u['completion']:,} completion in {u['calls']} model call(s)"
              + (f" (of which {u['reasoning']:,} reasoning)" if u.get("reasoning") else ""))
    if a.trace_id:
        print(f"Decision trail saved as trace #{a.trace_id}:  python -m kb trace {a.trace_id}")
    print("Sources:")
    for h in a.hits:
        tag = h["metadata"].get("table") if h["kind"] == "schema" else f"p.{h['page']}"
        print(f"  - {h['filename']} {tag} [{h['kind']}]" + (f"  image: {h['image_path']}" if h["image_path"] else ""))


def main():
    ap = argparse.ArgumentParser(
        prog="kb", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Knowledge base over PDF documentation, schema diagrams and a live MySQL schema, stored in pgvector.\n"
                    "Answers questions from the documentation or by writing and running read-only SQL.",
        epilog="typical flow:\n"
               "  python -m kb status --deep    check databases and models\n"
               "  python -m kb setup            create tables, load the MySQL schema, ingest ./pdfs\n"
               "  python -m kb ask \"...\"        answer a question\n"
               "  python -m kb trace ID         show how a question was answered\n\n"
               "run 'python -m kb <command> --help' for the options of one command.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create pgvector extension + tables")
    st = sub.add_parser("status", help="check pgvector, MySQL, vision model and LLM")
    st.add_argument("--deep", action="store_true", help="send one tiny real request to each model and show the provider error")
    s = sub.add_parser("setup", help="init + sync MySQL schema + ingest ./pdfs  (one-shot demo prep)")
    s.add_argument("--no-vision", action="store_true", help="skip the diagram model")
    sub.add_parser("sync-schema", help="store the live MySQL schema as searchable cards")
    i = sub.add_parser("ingest", help="ingest a PDF or a folder of PDFs")
    i.add_argument("path")
    i.add_argument("--no-vision", action="store_true")
    i.add_argument("--force", action="store_true")
    q = sub.add_parser("search", help="retrieval only (no LLM)")
    q.add_argument("question"); q.add_argument("-k", type=int, default=8)
    a = sub.add_parser("ask", help="retrieve, answer or write+run SQL")
    a.add_argument("question"); a.add_argument("-k", type=int, default=8)
    a.add_argument("--no-exec", action="store_true", help="show generated SQL but do not run it")
    u = sub.add_parser("usage", help="token usage per day / model / purpose, with warnings about truncation")
    u.add_argument("--days", type=int, default=7)
    vt = sub.add_parser("vision-test", help="run ONLY the diagram step on a PDF and print what the vision model says (nothing is stored)")
    vt.add_argument("pdf")
    vt.add_argument("--limit", type=int, default=1, help="how many figures to try")
    h = sub.add_parser("history", help="list recent questions and how they were answered")
    h.add_argument("-n", type=int, default=15)
    tr = sub.add_parser("trace", help="show the full decision trail (retrieval, prompts, model thinking, SQL tries) of one question")
    tr.add_argument("id", type=int)
    tr.add_argument("--full", action="store_true", help="do not shorten prompts")
    e = sub.add_parser("eval", help="retrieval hit-rate on eval/questions.json (no LLM)")
    e.add_argument("-k", type=int, default=8)
    args = ap.parse_args()
    from .config import setup_logging
    setup_logging()

    if args.cmd == "init":
        from .db import init_db; init_db(); print("pgvector ready")
    elif args.cmd == "status":
        from .status import run; sys.exit(run(args.deep))
    elif args.cmd == "setup":
        from .config import MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE
        from .db import init_db
        from .mysql_schema import wait_for_mysql
        from .ingest import sync_schema, ingest_path
        from .usage import run_summary
        t0 = time.perf_counter()
        init_db(); log.info("pgvector ready (tables: documents, chunks, query_log, llm_usage)")
        wait_for_mysql(); log.info("connected to MySQL %s:%s/%s", MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE)
        sync_schema()
        if pathlib.Path("pdfs").exists():
            ingest_path("pdfs", vision=not args.no_vision)
        log.info("done in %.0fs | %s", time.perf_counter() - t0, run_summary())
        print("\nNext:  python -m kb status   |   python -m kb ask \"Which 5 customers paid the most?\"   |   streamlit run app.py")
    elif args.cmd == "sync-schema":
        from .ingest import sync_schema; sync_schema(force=True)
    elif args.cmd == "ingest":
        from .ingest import ingest_path; ingest_path(args.path, vision=not args.no_vision, force=args.force)
    elif args.cmd == "search":
        from .retrieve import retrieve
        for h in retrieve(args.question, args.k):
            print(f"--- {h['filename']} p.{h['page']} [{h['kind']}] score={h['score']:.4f}\n{h['content'][:400]}\n")
    elif args.cmd == "ask":
        from .answer import ask; _print_answer(ask(args.question, args.k, execute=not args.no_exec))
    elif args.cmd == "vision-test":
        import io
        from PIL import Image
        from .pdf_parser import parse_pdf
        from .quality import looks_degenerate
        from .vision import describe_raw, VisionError
        _, pages, figures = parse_pdf(args.pdf, vision=True)
        if not figures:
            sys.exit("no figures found in this PDF")
        page_text = {n: "\n".join(lines) for n, lines in pages}
        pathlib.Path("page_images").mkdir(exist_ok=True)
        for i, fig in enumerate(figures[:args.limit], start=1):
            path = f"page_images/vision_test_p{fig.page}_{i}.png"
            pathlib.Path(path).write_bytes(fig.png)
            w, h = Image.open(io.BytesIO(fig.png)).size
            print(f"--- page {fig.page}, {fig.source} figure, {w}x{h}px, {len(fig.png) // 1024} KB, saved to {path}")
            try:
                out, info = describe_raw(fig.png, page_text[fig.page] if fig.source == "vector" else "")
            except VisionError as e:
                print(f"vision model error: {e}")
                continue
            print(f"call: {info}\ndegenerate (repeated lines): {looks_degenerate(out)}\n\n{out[:3500]}\n")
    elif args.cmd == "usage":
        from .usage import usage_rows, recent_flags, format_summary
        print(format_summary(usage_rows(args.days), recent_flags(), args.days))
    elif args.cmd == "history":
        from .trace import list_traces
        for r in list_traces(args.n):
            print(f"#{r['id']:<4} {str(r['created_at'])[:19]}  {r['outcome']:<18} {str(r['total_ms']):>6} ms  {r['question'][:70]}")
    elif args.cmd == "trace":
        from .trace import get_trace, format_trace
        t = get_trace(args.id)
        print(format_trace(t, args.full) if t else f"no trace #{args.id}")
    elif args.cmd == "eval":
        from .evaluate import run; sys.exit(run(k=args.k))


main()
