import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # keeps the pure-Python modules importable in minimal environments
    pass


def _clean(name: str, default: str) -> str:
    """Numeric/boolean settings only: ignore a trailing '# comment'. Some tools (e.g. the VS Code Python extension)
    load .env themselves and keep inline comments in the value."""
    return os.getenv(name, default).split("#", 1)[0].strip()


def _int(name: str, default: int) -> int:
    return int(_clean(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_clean(name, str(default)))


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = _int("MYSQL_PORT", 3307)
MYSQL_USER = os.getenv("MYSQL_USER", "sakila")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "p_ssW0rd")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sakila")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = _int("EMBED_DIM", 384)

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
VLM_DELAY_SEC = _float("VLM_DELAY_SEC", 1.0)
VLM_TEMPERATURE = _float("VLM_TEMPERATURE", 0.0)
VLM_MAX_TOKENS = _int("VLM_MAX_TOKENS", 1500)
VLM_FREQUENCY_PENALTY = _float("VLM_FREQUENCY_PENALTY", 0.0)  # 0 = not sent (some providers reject it)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")

DIAGRAM_MIN_DRAWINGS = _int("DIAGRAM_MIN_DRAWINGS", 40)
IMAGE_DIR = os.getenv("IMAGE_DIR", "page_images")
RETRIEVE_K = _int("RETRIEVE_K", 8)
MAX_ROWS = _int("MAX_ROWS", 100)

# Decision-trail log (see kb/trace.py). Prompts, retrieved text and result rows are stored, so keep the DB local.
TRACE_ENABLED = _clean("TRACE_ENABLED", "1").lower() not in ("0", "false", "no", "")
TRACE_MAX_CHARS = _int("TRACE_MAX_CHARS", 30000)
