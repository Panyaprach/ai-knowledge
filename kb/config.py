import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3307"))
MYSQL_USER = os.getenv("MYSQL_USER", "sakila")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "p_ssW0rd")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sakila")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:7b")
VLM_DELAY_SEC = float(os.getenv("VLM_DELAY_SEC", "1.0"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")

DIAGRAM_MIN_DRAWINGS = int(os.getenv("DIAGRAM_MIN_DRAWINGS", "40"))
IMAGE_DIR = os.getenv("IMAGE_DIR", "page_images")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))
