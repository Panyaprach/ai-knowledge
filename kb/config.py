import logging
import logging.handlers
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _clean(name: str, default: str) -> str:
    """Parse numeric/boolean settings, ignoring a trailing '# comment'."""
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
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "").strip()
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip()
EMBED_SEND_DIMENSIONS = _clean("EMBED_SEND_DIMENSIONS", "0").lower() in ("1", "true", "yes")


def _prefix(name: str) -> str:
    v = os.getenv(name, "")
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" else v


EMBED_QUERY_PREFIX = _prefix("EMBED_QUERY_PREFIX")
EMBED_DOC_PREFIX = _prefix("EMBED_DOC_PREFIX")

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
VLM_DELAY_SEC = _float("VLM_DELAY_SEC", 1.0)
VLM_TEMPERATURE = _float("VLM_TEMPERATURE", 0.0)
VLM_MAX_TOKENS = _int("VLM_MAX_TOKENS", 1500)
VLM_FREQUENCY_PENALTY = _float("VLM_FREQUENCY_PENALTY", 0.0)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")

DIAGRAM_MIN_DRAWINGS = _int("DIAGRAM_MIN_DRAWINGS", 40)
IMAGE_DIR = os.getenv("IMAGE_DIR", "page_images")
RETRIEVE_K = _int("RETRIEVE_K", 8)
MAX_ROWS = _int("MAX_ROWS", 100)

TRACE_ENABLED = _clean("TRACE_ENABLED", "1").lower() not in ("0", "false", "no", "")
TRACE_MAX_CHARS = _int("TRACE_MAX_CHARS", 30000)
USAGE_ENABLED = _clean("USAGE_ENABLED", "1").lower() not in ("0", "false", "no", "")


class _Compact(logging.Formatter):
    """[12:01:07] vision   message      (warnings and errors get a level word: [12:01:07] WARNING vision   message)"""
    def __init__(self):
        super().__init__("[%(asctime)s] %(lvl)s%(name)-8s %(message)s", "%H:%M:%S")

    def format(self, record):
        r = logging.makeLogRecord(record.__dict__)
        r.name = r.name.removeprefix("kb.")
        r.lvl = r.levelname + " " if r.levelno >= logging.WARNING else ""
        return super().format(r)


def setup_logging(level: str | None = None) -> None:
    """Configure the 'kb' logger from KB_LOG_LEVEL and KB_LOG_FILE. Safe to call repeatedly."""
    name = (level or os.getenv("KB_LOG_LEVEL", "info")).split("#", 1)[0].strip().upper()
    lvl = logging.WARNING if name == "QUIET" else getattr(logging, name, logging.INFO)
    if not isinstance(lvl, int):
        lvl = logging.INFO
    root = logging.getLogger("kb")
    root.setLevel(lvl)
    root.propagate = False
    if not any(getattr(h, "_kb", False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(_Compact())
        console._kb = True
        root.addHandler(console)
        path = os.getenv("KB_LOG_FILE", "").split("#", 1)[0].strip()
        if path:
            f = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            f.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            f._kb = True
            root.addHandler(f)
