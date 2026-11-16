# Sakila knowledge base

Ingests a PDF (text and schema diagrams) and the live schema of a MySQL database into PostgreSQL/pgvector, then answers
questions about the database. Documentation questions are answered from the retrieved text; data questions are answered by
writing a read-only SQL query and running it on MySQL. Embeddings, the vision model and the answer model can each run
locally or through any OpenAI-compatible endpoint. The bundled example uses the Sakila sample database and its manual.

## How it works

```
PDF text, split by section ----------------------+
PDF figures -> vision model -> text -------------+--> embeddings --> pgvector (chunks)
MySQL information_schema -> one card per table --+

question -> hybrid search (vector + full-text) -> tables named in the question and their foreign-key neighbours
         -> model: answer from the documentation, or write SQL -> validate -> run on MySQL (read-only) -> answer
```

Every question is saved with its retrieved chunks, prompts, SQL attempts and token counts (`query_log`, `llm_usage`).

## Prerequisites

- Docker with Compose
- Python 3.10 or newer
- A vision-capable model and a text model behind an OpenAI-compatible API, for example Ollama running locally or a hosted
  provider with an API key
- A local embedding model is downloaded on first use unless `EMBED_BASE_URL` points to an embeddings API

## Quickstart

```bash
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to point `VLM_*` and `LLM_*` at your models. With Ollama: `ollama pull qwen2.5vl:7b` and
`ollama pull qwen2.5-coder:7b`. Then:

```bash
python -m kb status --deep
python -m kb setup
python -m kb ask "How many films are in each category?"
streamlit run app.py
```

`python -m kb setup --no-vision` skips the diagram step if no vision model is available.

## Usage

```bash
python -m kb --help
python -m kb <command> --help
```

| Command | Purpose |
|---|---|
| `status` | check the databases and models (`--deep` sends a test request to each) |
| `setup`, `init`, `sync-schema`, `ingest` | prepare the knowledge base |
| `ask`, `search` | answer a question, or show retrieval only |
| `history`, `trace`, `usage` | review past questions, their decision trail and token usage |
| `eval` | check retrieval against `eval/questions.json` |
| `vision-test` | run only the diagram step on a PDF |

The Streamlit app has three pages: the question page, **History** (decision trail per question) and **Usage** (token counts).
`make` shortcuts: `up`, `setup`, `demo`, `status`, `eval`, `test`, `down`.

Adminer runs at http://localhost:8080:

| System | Server | User | Password | Database |
|---|---|---|---|---|
| PostgreSQL | `pgvector` | `rag` | `rag` | `rag` |
| MySQL | `mysql` | `sakila` | `p_ssW0rd` | `sakila` |

## Configuration

Settings are read from `.env` (see `.env.example`).

| Group | Variables |
|---|---|
| Databases | `DATABASE_URL`, `MYSQL_*` |
| Embeddings | `EMBED_MODEL`, `EMBED_DIM`, `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_SEND_DIMENSIONS`, `EMBED_QUERY_PREFIX`, `EMBED_DOC_PREFIX` |
| Vision model | `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL`, `VLM_TEMPERATURE`, `VLM_MAX_TOKENS` |
| Answer model | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` |
| Retrieval | `RETRIEVE_K`, `MAX_ROWS`, `DIAGRAM_MIN_DRAWINGS` |
| Logging | `KB_LOG_LEVEL`, `KB_LOG_FILE`, `TRACE_ENABLED`, `USAGE_ENABLED` |

Embeddings run locally with sentence-transformers when `EMBED_BASE_URL` is empty. Otherwise they use `/embeddings` on that
endpoint. `EMBED_DIM` must equal the model's vector size, and the vector table is created with that size, so after changing
the embedding model or size recreate it (`docker compose down -v`) and run `setup` again. Some models expect different
prefixes for queries and documents; set `EMBED_QUERY_PREFIX` and `EMBED_DOC_PREFIX` for those.

To use Oracle's official Sakila files instead of the bundled image, run `bash scripts/fetch_sakila.sh`, start with
`docker compose -f docker-compose.yml -f docker-compose.official-mysql.yml up -d`, and set `MYSQL_USER=root`,
`MYSQL_PASSWORD=root`.

## Project layout

```
kb/            application code (python -m kb)
  __main__.py    command-line interface
  ingest.py      PDF and schema ingestion      pdf_parser.py, chunking.py, vision.py
  embed.py       local or API embeddings       retrieve.py, linking.py
  answer.py      question answering            sqlguard.py, mysql_schema.py
  trace.py       decision trail                usage.py, status.py, evaluate.py
app.py, pages/ Streamlit app
eval/          retrieval test questions
pdfs/          documents to ingest
tests/         unit tests
```

## Development

```bash
pip install -r requirements.txt
make test
```

Generated SQL is validated with sqlglot (one `SELECT`, known tables only, capped `LIMIT`) and runs in a read-only session. Use
a MySQL account with only `SELECT` for anything beyond a local demo.
