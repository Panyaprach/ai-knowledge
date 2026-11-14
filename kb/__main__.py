import argparse, pathlib, sys


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
    print("Sources:")
    for h in a.hits:
        tag = h["metadata"].get("table") if h["kind"] == "schema" else f"p.{h['page']}"
        print(f"  - {h['filename']} {tag} [{h['kind']}]" + (f"  image: {h['image_path']}" if h["image_path"] else ""))


def main():
    ap = argparse.ArgumentParser(prog="kb", description="Sakila knowledge base: PDF + diagram + live schema in pgvector, answered with RAG and text-to-SQL")
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
    e = sub.add_parser("eval", help="retrieval hit-rate on eval/questions.json (no LLM)")
    e.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    if args.cmd == "init":
        from .db import init_db; init_db(); print("pgvector ready")
    elif args.cmd == "status":
        from .status import run; sys.exit(run(args.deep))
    elif args.cmd == "setup":
        from .db import init_db
        from .mysql_schema import wait_for_mysql
        from .ingest import sync_schema, ingest_path
        init_db(); print("pgvector ready")
        wait_for_mysql(); sync_schema()
        if pathlib.Path("pdfs").exists():
            ingest_path("pdfs", vision=not args.no_vision)
        print("\nDone. Try:  python -m kb ask \"Which 5 customers paid the most?\"   or   streamlit run app.py")
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
    elif args.cmd == "eval":
        from .evaluate import run; sys.exit(run(k=args.k))


main()
