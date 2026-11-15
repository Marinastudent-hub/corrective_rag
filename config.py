
import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.groq.com/openai/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# Semantic embedding model
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 10
MAX_RETRIES = 2
