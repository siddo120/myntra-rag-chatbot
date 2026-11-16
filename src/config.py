"""Central config + env loading for the Myntra RAG chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local secrets file (git-ignored). On Streamlit Cloud the key comes from st.secrets.
load_dotenv(PROJECT_ROOT / "kaggle-rag-chatbot.env")
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Fashion Dataset.csv"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "products_clean.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "myntra_products"

# --- Embeddings: local model, no API key, runs on CPU ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim
EMBED_BATCH = 256

# --- LLM: any OpenAI-compatible endpoint. Groq by default, OpenAI if only that key is set ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_TIMEOUT_S = 20.0
LLM_MAX_RETRIES = 2


def llm_settings() -> dict:
    """Resolve key / endpoint / model at call time (st.secrets is bridged into env late)."""
    groq, oai = os.getenv("GROQ_API_KEY"), os.getenv("OPENAI_API_KEY")
    if groq:
        key, base, model = groq, GROQ_BASE_URL, "openai/gpt-oss-120b"
    elif oai:
        key, base, model = oai, None, "gpt-4o-mini"
    else:
        raise MissingKeyError(
            "No LLM key found. Set GROQ_API_KEY (or OPENAI_API_KEY) in kaggle-rag-chatbot.env "
            "or in Streamlit secrets."
        )
    base = os.getenv("LLM_BASE_URL", base)
    model = os.getenv("LLM_MODEL", model)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    return {"api_key": key, "base_url": base, "model": model, "extra": extra}


class MissingKeyError(RuntimeError):
    pass


# --- Retrieval ---
TOP_K = 5
CANDIDATE_POOL = 30          # per retriever, before fusion
AGGREGATE_POOL = 60          # candidates considered for "cheapest / best rated" questions
RRF_K = 60
SEMANTIC_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.6
MIN_RATINGS_FOR_BEST = 10    # "best rated" ignores products with fewer ratings than this

# Relevance gate (cosine distance, calibrated for EMBED_MODEL; see eval/gate_analysis.py).
# A query is off-topic when the nearest product is farther than this AND no parsed
# constraint (brand / category / colour) ties it to the catalogue.
RELEVANCE_MAX_DISTANCE = 0.39

# --- Conversation ---
HISTORY_TURNS = 3            # previous user/assistant pairs given to the query rewriter
