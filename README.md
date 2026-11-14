# Sakila knowledge base — PDF + diagram + live schema in pgvector, answered with RAG and text-to-SQL

Two databases, two jobs:

| | Role | Runs as |
|---|---|---|
| **MySQL (Sakila)** | the *business* database: real DVD-rental data | `mysql` service in Docker Compose |
| **PostgreSQL + pgvector** | the *vector* database: the knowledge base | `pgvector` service in Docker Compose |

```
 sakila-en_a4.pdf ──► text, split by section ("5.1.6 The customer Table") ─────────────┐
        │                                                                              │
        └─► Figure 1 (schema diagram) ─► crop at native resolution ─► vision model ─► text   ├─► embed (local) ─► pgvector
                                                                                       │      (kind = text | diagram | schema)
 MySQL information_schema ─► one "schema card" per table/view (columns, types, FKs) ───┘
                                                  ▲
 question ─► hybrid search (vector + full-text, RRF) ─► + FK-neighbour tables (graph expansion)
          ─► LLM decides:  docs question → answer with [file p.N] citations
                           data question → write SELECT → guard → run on MySQL (READ ONLY) → self-repair on error → summarise
```

Why this shape: the PDF explains *meaning* (e.g. "`active` is an alternative to deleting a customer"), the diagram gives an
*overview*, and the schema cards give the *exact* column names the SQL needs. Retrieval finds the tables a question is about, then
pulls in the tables joined to them by foreign key, so multi-hop questions ("films per category" → `film_category`) get their bridge table.

## Run the demo

```bash
docker compose up -d                       # pgvector + MySQL (Sakila preloaded)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# free local models (skip if you use Hugging Face / AI Studio keys in .env):
docker compose --profile ollama up -d
docker compose exec ollama ollama pull qwen2.5-coder:7b     # answers + SQL
docker compose exec ollama ollama pull qwen2.5vl:7b         # reads the diagram (lighter: qwen2.5vl:3b)

python -m kb status                        # which services are up?
python -m kb setup                         # init → load schema cards → ingest pdfs/sakila-en_a4.pdf
streamlit run app.py                       # the demo UI
```
Weak laptop / no vision model? `python -m kb setup --no-vision` still works: you lose only the diagram chunk;
the schema cards already carry every FK.

### Suggested demo script (≈5 min)
1. *"Which table links films and actors?"* → answered from the PDF, cited with page.
2. *"What does the active column in customer mean?"* → business rule from the docs.
3. *"What is the known design issue with stores and staff?"* → "Known Issues" section.
4. *"How many films are in each category?"* → writes SQL joining `film_category`; open **Sources** to show the graph-expanded bridge table.
5. *"Who are the 5 customers who paid the most in total?"* → SQL + result table + summary.
6. Open a **diagram** source: the original Figure 1 image shown next to the text the model transcribed from it.
7. Try *"delete all rentals"* → the guard refuses; the session is READ ONLY anyway.

CLI equivalents: `python -m kb ask "..."`, `python -m kb search "..."`, `python -m kb ask "..." --no-exec`.

## Browse the data (Adminer)
`docker compose up -d adminer` then open http://localhost:8080 . Pick the system in the login form:

| | System | Server | Username | Password | Database |
|---|---|---|---|---|---|
| vector DB | PostgreSQL | `pgvector` | `rag` | `rag` | `rag` |
| Sakila | MySQL | `mysql` | `sakila` | `p_ssW0rd` | `sakila` |

Look at `chunks` (columns `kind`, `page`, `content`, `metadata`, `image_path`) and `documents` in pgvector. The `embedding`
column is a long vector, so select the other columns instead of `*`. Adminer is bound to localhost only; don't expose it.

## Quality checks
- `make test` — unit tests (SQL guard, header/footer/TOC cleanup, heading-aware chunking). No services needed.
- `python -m kb eval` — retrieval hit-rate over `eval/questions.json` (no LLM needed). Edit the file to add your own questions.

## If the MySQL image doesn't work for you
The default `sakiladb/mysql:8` is a community image with Sakila baked in (user `sakila`, password `p_ssW0rd`).
To use Oracle's official files instead:
```bash
bash scripts/fetch_sakila.sh
docker compose -f docker-compose.yml -f docker-compose.official-mysql.yml up -d
# .env: MYSQL_USER=root  MYSQL_PASSWORD=root   (first start takes a minute while data loads)
```

## Free model options (edit `.env`, no code change)
Everything talks to OpenAI-compatible endpoints: Ollama (local, free), Hugging Face Inference Providers, Google AI Studio.
Hosted free tiers have rate limits and change their model lists, so check them; the vision client retries with backoff.

## Safety of text-to-SQL
Layered: (1) validator allows one `SELECT`/`WITH` only, blocks writes, comments, `INTO OUTFILE`, `SLEEP`, etc.;
(2) session forced `READ ONLY` with a 10 s limit; (3) at most 100 rows returned. For anything beyond a demo, also give the
app a MySQL user that only has `SELECT`.

## Known limits
- Not run end to end in the authoring sandbox (no network/Docker). The chunker was checked against the real Sakila PDF text
  (62 section-aware chunks, headers/footers/TOC/legal boilerplate removed); DB, MySQL and model calls are untested here,
  so expect small first-run fixes.
- Figure 1 shows table names and relationship labels, not columns, and its text is small. 7B vision models may miss lines or
  cardinalities; that is why schema cards exist. A stronger model for the one-time ingest is the best upgrade.
- Small local models write imperfect SQL; the self-repair loop (up to 2 retries) helps but doesn't make it perfect. Prefer a
  coder model for `LLM_MODEL`.
- Changing `EMBED_MODEL` to a different dimension needs a fresh pgvector volume (`docker compose down -v`).
- Scanned PDFs need OCR (not included).
