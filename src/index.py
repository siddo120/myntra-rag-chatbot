"""Index lifecycle: one build path, a manifest, and startup verification.

The vector index (ChromaDB), the keyword index (BM25) and the product table are
all derived from the same products_clean.csv. The collection's metadata records
which CSV (sha256), how many rows and which embedding model it was built from;
`load()` refuses to serve an index that does not match and rebuilds it instead.

Builds are atomic: vectors go into a staging collection that only replaces the
live one after every row is in, so a crash mid-build can never leave a partial
index being served.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass

# Streamlit Cloud ships an old system sqlite3 that ChromaDB rejects. If the
# pysqlite3 backport is installed, swap it in before importing chromadb.
try:  # pragma: no cover
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import pandas as pd
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

import config
from textutil import tokenize

log = logging.getLogger("myntra.index")
# chromadb 0.5.x logs an error per call from its (disabled) telemetry client.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
_STAGING = config.COLLECTION_NAME + "__staging"
_lock = threading.Lock()
_embedder = None
_state = None


# ---------------------------------------------------------------- embeddings ---
def _model():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(config.EMBED_MODEL)
    return _embedder


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts, batch_size=64)]


def embed_query(text: str) -> list[float]:
    return next(iter(_model().query_embed(text))).tolist()


# ------------------------------------------------------------------ manifest ---
def csv_sha256(path=None) -> str:
    h = hashlib.sha256()
    with open(path or config.PROCESSED_CSV, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def _metadata(row) -> dict:
    return {
        "name": str(row["name"]), "brand": str(row["brand"]), "colour": str(row["colour"]),
        "category": str(row["category"]), "gender": str(row["gender"]),
        "price": float(row["price"]),
        "avg_rating": float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else -1.0,
        "rating_count": int(row["rating_count"]) if pd.notna(row["rating_count"]) else 0,
    }


def read_products() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_CSV, keep_default_na=False,
                     na_values={"avg_rating": [""], "rating_count": [""]})
    df["id"] = df["product_id"].astype(int).astype(str)
    return df


def is_current(collection, df, sha) -> bool:
    m = collection.metadata or {}
    return (m.get("csv_sha256") == sha and m.get("embed_model") == config.EMBED_MODEL
            and m.get("rows") == len(df) and collection.count() == len(df))


def build(embed=embed_documents, progress=None):
    """Embed every product into a staging collection, then swap it in."""
    df = read_products()
    sha = csv_sha256()
    client = _client()
    for name in (_STAGING,):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    # The manifest is written up front; a collection only counts as current when its
    # row count also matches, and staging is renamed to the live name only when complete.
    staging = client.create_collection(_STAGING, metadata={
        "hnsw:space": "cosine", "csv_sha256": sha, "embed_model": config.EMBED_MODEL, "rows": len(df)})
    for start in range(0, len(df), config.EMBED_BATCH):
        chunk = df.iloc[start:start + config.EMBED_BATCH]
        texts = chunk["retrieval_text"].tolist()
        staging.add(ids=chunk["id"].tolist(), embeddings=embed(texts), documents=texts,
                    metadatas=[_metadata(r) for _, r in chunk.iterrows()])
        if progress:
            progress(min(start + len(chunk), len(df)), len(df))
    if staging.count() != len(df):
        raise RuntimeError(f"index build incomplete: {staging.count()} of {len(df)} rows")
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    staging.modify(name=config.COLLECTION_NAME)
    return client.get_collection(config.COLLECTION_NAME)


# --------------------------------------------------------------------- state ---
@dataclass
class State:
    collection: object
    products: pd.DataFrame        # indexed by product id (str)
    bm25: BM25Okapi
    tokens: list                  # per product: set of stemmed tokens (same order as ids)
    ids: list
    brands: list
    colours: list


def _bm25_text(row) -> str:
    # Name carries the strongest signal; repeat it so it outweighs the description.
    return f"{row['name']} {row['name']} {row['brand']} {row['category']} {row['attributes']} {row['description']}"


def load(progress=None) -> State:
    """Return the verified, shared index state (built once per process)."""
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        df = read_products()
        sha = csv_sha256()
        try:
            collection = _client().get_collection(config.COLLECTION_NAME)
            ok = is_current(collection, df, sha)
        except Exception:
            collection, ok = None, False
        if not ok:
            log.warning("vector index missing or out of date; rebuilding %d products", len(df))
            collection = build(progress=progress)
        corpus = [tokenize(_bm25_text(r)) for _, r in df.iterrows()]
        _state = State(collection=collection, products=df.set_index("id", drop=False),
                       bm25=BM25Okapi(corpus), tokens=[set(t) for t in corpus], ids=df["id"].tolist(),
                       brands=sorted(df["brand"].unique(), key=len, reverse=True),
                       colours=sorted(df["colour"].unique(), key=len, reverse=True))
        return _state


def reset() -> None:
    """Drop the cached state (tests, or after a rebuild)."""
    global _state
    with _lock:
        _state = None
