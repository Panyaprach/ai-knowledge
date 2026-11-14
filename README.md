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

## Observe it: the decision trail
Every question is saved in the `query_log` table (one row, ordered `events`) and can be read three ways:
- **Streamlit:** the **History** page in the sidebar (pick a question; each step is laid out in order).
- **Terminal:** `python -m kb history` then `python -m kb trace 12` (add `--full` to see complete prompts).
- **Adminer:** table `query_log`, column `events` (JSON).

What a trail contains, in the order things happen:

| Stage | What it tells you |
|---|---|
| `retrieve` | which chunks came back from pgvector (text / diagram / schema), their score, and which were *added via foreign key* instead of found by search |
| `llm_call` | the exact prompt, the raw reply, **the model's thinking if the provider returned any**, `finish_reason`, token counts, time. Purposes: *route* (docs answer or SQL?), *repair* (after a failed query), *summarise* (turn rows into a sentence) |
| `decision` | the route taken and why: docs (reply had no ```sql block) or SQL |
| `sql_attempt` | each try: **rejected by the SQL guard** / **MySQL error** / **ok**, the SQL in, the SQL that actually ran (LIMIT added), a 5-row preview. A failed try's error is what the model sees when it repairs |
| `error` | an exception that stopped the run (for example the model server being down) |

How "thinking" is captured: it is read from a separate reply field (`reasoning_content`, `reasoning` or `thinking`, the names
different providers use) and from `<think>...</think>` inside the reply. If the provider returns neither, the page says so.
It is the model's own account, not proof of how it decided, so use the decision events as the ground truth.

Reading tips:
- `finish_reason=length` means the reply was cut off by `max_tokens`. Thinking models can spend the whole budget on thinking and
  return an empty or partial reply; raise the limit in `kb/answer.py` (`_chat(... max_tokens=...)`) if you see it.
- Two `llm_call`s plus a `sql_attempt` per question is normal for data questions; a docs question has only one `llm_call`.
- Logging never breaks the app: if saving fails it prints one line and carries on. Set `TRACE_ENABLED=0` to turn it off.
- The log holds prompts, retrieved text and a 5-row preview of results. Keep the database local.

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
Layered, and no longer keyword-based: `kb/sqlguard.py` parses the generated SQL with **sqlglot** (MySQL dialect) and checks the
syntax tree. It allows one `SELECT` (CTEs and UNION are fine) and rejects any write/DDL node, `SELECT ... INTO`, `FOR UPDATE`,
`@variables`, dangerous functions (`SLEEP`, `BENCHMARK`, `LOAD_FILE`, ...) and comments. Every table must be a real table of the
Sakila database, so `information_schema` and `mysql.*` are unreachable. A `LIMIT` is always added and capped, and the SQL that
runs is regenerated from the tree (shown in the UI). Then the session is forced `READ ONLY` with a 10 s limit. Rejection messages
list the available tables so the model can repair its query. For anything beyond a demo, also use a MySQL account with only `SELECT`.

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
