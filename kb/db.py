import psycopg
from pgvector.psycopg import register_vector
from .config import DATABASE_URL, EMBED_DIM
from .trace import QUERY_LOG_DDL  # defined next to the code that writes it

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,           -- 'x.pdf' or 'mysql://sakila'
    sha256      TEXT NOT NULL UNIQUE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INT NOT NULL,
    kind        TEXT NOT NULL,           -- 'text' | 'diagram' | 'schema'
    content     TEXT NOT NULL,
    image_path  TEXT,                    -- PNG of the figure for diagram chunks
    embedding   vector({EMBED_DIM}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    metadata    JSONB NOT NULL DEFAULT '{{}}'
);
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_kind_check;  -- older version of this project had one
CREATE INDEX IF NOT EXISTS chunks_emb_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin  ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_page ON chunks (document_id, page);
CREATE INDEX IF NOT EXISTS chunks_meta_table ON chunks ((metadata->>'table'));
"""


def init_db():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA)
        conn.execute(QUERY_LOG_DDL)


def connect():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn


def chunk_counts() -> dict:
    with connect() as conn:
        return dict(conn.execute("SELECT kind, count(*) FROM chunks GROUP BY kind").fetchall())
