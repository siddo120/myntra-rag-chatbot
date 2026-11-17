"""
RAG core for the Myntra fashion chatbot.

Retrieval modes
---------------
- semantic : OpenAI query embedding -> ChromaDB cosine search (meaning-based)
- hybrid   : semantic  +  BM25 keyword search, fused with Reciprocal Rank Fusion
             (catches exact terms like brand names / fabrics that pure semantic
             search can miss, while keeping meaning-based matching)

Then: relevance gate -> grounded gpt-4o-mini answer (or fallback).

Public API:
    answer(query, hybrid=False, ...) -> dict(answer, sources, used_fallback, retrieved)
    retrieve(query, ...)             -> semantic hits
    retrieve_hybrid(query, ...)      -> fused hits
"""

import re
from functools import lru_cache

# Streamlit Cloud ships an old system sqlite3 that ChromaDB rejects. If the
# pysqlite3 backport is installed, swap it in before importing chromadb.
try:  # pragma: no cover
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass  # local machine has a new-enough sqlite3

import chromadb
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi

import config
import query_understanding as qu

# If even the closest product is farther than this cosine distance, treat the
# question as out-of-catalogue and fall back instead of forcing an answer.
RELEVANCE_MAX_DISTANCE = 0.72

# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 20

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for a Myntra-style fashion catalogue "
    "(mostly women's apparel). Answer the user's question USING ONLY the product "
    "context provided below. Follow these rules strictly:\n"
    "1. Ground every claim (product name, brand, price, fabric, colour, fit, "
    "occasion, rating) in the given context. Never invent products or details.\n"
    "2. If the context does not contain enough information to answer, say so "
    "clearly (e.g. \"I couldn't find that in the catalogue\") and suggest how the "
    "user might rephrase or broaden their search. Do NOT guess.\n"
    "3. When recommending products, cite the product name, brand and price (in INR).\n"
    "4. Be concise and friendly. Use a short list when suggesting multiple items."
)

FILTERED_SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for a Myntra-style women's & girls' "
    "fashion catalogue. The products in the context below have ALREADY been "
    "filtered to match the user's required constraints (type, colour, price, "
    "fabric). PRESENT them:\n"
    "1. Recommend from these products — they already satisfy the hard "
    "requirements, so do NOT say you couldn't find anything.\n"
    "2. Cite product name, brand, price (INR), colour and rating.\n"
    "3. If a softer preference (occasion/festive/style) isn't explicitly "
    "confirmed in a product's text, you may note that briefly, but still "
    "present the product.\n"
    "4. Never invent products or details beyond the context. Be concise; use a "
    "short list for multiple items."
)

FALLBACK_MESSAGE = (
    "I couldn't find anything relevant in the Myntra catalogue for that. "
    "This catalogue is mostly women's fashion (kurtas, dresses, tops, ethnic & "
    "western wear). Try rephrasing, or ask about a product type, fabric, colour, "
    "occasion or price range."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


# ---------------------------------------------------------------- clients ----
@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    config.require_openai_key()
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


@lru_cache(maxsize=1)
def _collection():
    """Return the ChromaDB collection, building it from the cleaned CSV if empty.

    This makes the app self-bootstrapping: on a fresh host (e.g. Streamlit Cloud)
    where only the cleaned CSV is shipped, the vector index is rebuilt on first
    use with the SAME chunking + embeddings as the local build.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if col.count() > 0:
        return col

    # ---- build on startup ----
    oai = _openai()
    df = pd.read_csv(config.PROCESSED_CSV)
    batch = config.EMBED_BATCH
    for start in range(0, len(df), batch):
        chunk = df.iloc[start:start + batch]
        vectors = _embed_batch(oai, chunk["retrieval_text"].astype(str).tolist())
        col.add(
            ids=[str(int(r["product_id"])) for _, r in chunk.iterrows()],
            embeddings=vectors,
            documents=chunk["retrieval_text"].astype(str).tolist(),
            metadatas=[{
                "name": str(r["name"]), "brand": str(r["brand"]),
                "colour": str(r["colour"]),
                "price": float(r["price"]) if pd.notna(r["price"]) else -1.0,
                "avg_rating": float(r["avg_rating"]) if pd.notna(r["avg_rating"]) else -1.0,
            } for _, r in chunk.iterrows()],
        )
    return col


@lru_cache(maxsize=1)
def _img_map() -> dict:
    """Map product_id -> image URL, read from the cleaned CSV (for the UI)."""
    if not config.PROCESSED_CSV.exists():
        return {}
    df = pd.read_csv(config.PROCESSED_CSV, usecols=["product_id", "img"]).dropna(subset=["product_id"])
    return {str(int(pid)): url for pid, url in zip(df["product_id"], df["img"])}


def ensure_index():
    """Trigger the (cached) build-if-empty. Call once at app startup."""
    col = _collection()
    return col.count()


@lru_cache(maxsize=1)
def _bm25_index():
    """Build an in-memory BM25 keyword index over the same corpus Chroma holds.

    Returns (bm25, ids, records) where records[id] = {document, metadata}.
    """
    df = pd.read_csv(config.PROCESSED_CSV)
    ids, corpus_tokens, records = [], [], {}
    for _, row in df.iterrows():
        _id = str(int(row["product_id"]))
        text = str(row["retrieval_text"])
        ids.append(_id)
        corpus_tokens.append(_tokenize(text))
        records[_id] = {
            "document": text,
            "metadata": {
                "name": str(row["name"]),
                "brand": str(row["brand"]),
                "colour": str(row["colour"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else -1.0,
                "avg_rating": float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else -1.0,
            },
        }
    return BM25Okapi(corpus_tokens), ids, records


def _passes_filters(meta: dict, price_max, colour) -> bool:
    if price_max is not None and not (0 <= meta.get("price", -1) <= float(price_max)):
        return False
    if colour and meta.get("colour", "").lower() != colour.lower():
        return False
    return True


# -------------------------------------------------- structured catalogue ----
@lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    """The cleaned CSV augmented with derived columns for structured filtering."""
    df = pd.read_csv(config.PROCESSED_CSV)
    df["product_id"] = df["product_id"].astype("Int64")
    df["id"] = df["product_id"].astype(str)
    df["name_low"] = df["name"].fillna("").str.lower()
    # brand-stripped name: avoids brand words (e.g. "Jaipur Kurti") being read
    # as a product type. Remove the brand prefix before type matching.
    df["name_notbrand"] = [
        nm.replace(br, " ", 1) if isinstance(br, str) and br else nm
        for nm, br in zip(df["name_low"], df["brand"].fillna("").str.lower())
    ]
    attrs = df["attributes"].fillna("")
    df["occasion"] = attrs.str.extract(r"Occasion:\s*([^.]+)")[0].str.strip()
    df["fabric_text"] = (attrs + " " + df["description"].fillna("")).str.lower()
    nm = df["name_low"]
    # flag men's items (rare) so women/girls queries can drop them
    df["is_mens"] = (nm.str.contains(r"\bmen\b|\bman\b|\bmale\b|\bgents\b", regex=True)
                     & ~nm.str.contains("women", regex=True))
    return df


def _type_mask(df: pd.DataFrame, canon_type: str) -> pd.Series:
    pat = qu.TYPE_PATTERNS.get(canon_type)
    if not pat:
        return pd.Series(True, index=df.index)
    return df["name_notbrand"].str.contains(pat, regex=True, na=False)


def structured_filter(pq: dict, mentions_men: bool = False) -> pd.DataFrame:
    """Apply all parsed hard filters to the catalogue; return the matching rows."""
    df = _catalog()
    mask = pd.Series(True, index=df.index)
    if pq.get("type"):
        mask &= _type_mask(df, pq["type"])
    inc = pq.get("colour_include") or set()
    exc = pq.get("colour_exclude") or set()
    if inc:
        mask &= df["colour"].isin(inc)
    if exc:
        mask &= ~df["colour"].isin(exc)
    if pq.get("price_max") is not None:
        mask &= (df["price"] >= 0) & (df["price"] <= pq["price_max"])
    if pq.get("price_min") is not None:
        mask &= df["price"] >= pq["price_min"]
    if pq.get("fabrics"):
        fab = "|".join(re.escape(f) for f in pq["fabrics"])
        mask &= df["fabric_text"].str.contains(fab, regex=True, na=False)
    # NOTE: occasion is deliberately NOT a hard filter. The data labels most
    # ethnic/festive items "Casual"/"Daily" (Casual 2614 vs Festive 393), so a
    # hard occasion filter wrongly discards them. "festive" instead influences
    # ranking via the semantic query. See structured_filter docstring.
    if pq.get("with_dupatta"):
        mask &= (df["name_low"].str.contains("dupatta")
                 | df["fabric_text"].str.contains("with dupatta"))
    # Scope: the catalogue is women's + girls'/kids'. If the query explicitly
    # asks for men's, REQUIRE men's items (honestly few exist); otherwise drop
    # the handful of men's items from the women/girls/kids pool.
    if mentions_men:
        mask &= df["is_mens"]
    else:
        mask &= ~df["is_mens"]
    return df[mask]


# ------------------------------------------------------------- retrievers ----
def embed_query(query: str) -> list[float]:
    resp = _openai().embeddings.create(model=config.EMBED_MODEL, input=[query])
    return resp.data[0].embedding


def retrieve(query: str, k: int = config.TOP_K,
             price_max: float | None = None,
             colour: str | None = None) -> list[dict]:
    """Semantic retrieval: k most similar products (optional metadata filter)."""
    conds = []
    if price_max is not None:
        conds.append({"price": {"$lte": float(price_max)}})
    if colour:
        conds.append({"colour": colour})
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)

    res = _collection().query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=where,
    )
    hits = []
    for doc, meta, dist, _id in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0], res["ids"][0]
    ):
        hits.append({"id": _id, "document": doc, "metadata": meta,
                     "distance": dist, "source": "semantic"})
    return hits


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge several ranked id-lists into one ranking."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, _id in enumerate(lst):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve_hybrid(query: str, k: int = config.TOP_K,
                    price_max: float | None = None,
                    colour: str | None = None) -> list[dict]:
    """Hybrid retrieval: fuse semantic + BM25 keyword results with RRF."""
    bm25, ids, records = _bm25_index()

    # 1. semantic candidates (already filtered by Chroma `where`)
    sem_hits = retrieve(query, k=CANDIDATE_POOL, price_max=price_max, colour=colour)
    sem_ids = [h["id"] for h in sem_hits]
    sem_dist = {h["id"]: h["distance"] for h in sem_hits}

    # 2. BM25 candidates, then apply the same metadata filter
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    bm25_ids = []
    for i in ranked:
        _id = ids[i]
        if scores[i] <= 0:
            break
        if _passes_filters(records[_id]["metadata"], price_max, colour):
            bm25_ids.append(_id)
        if len(bm25_ids) >= CANDIDATE_POOL:
            break

    # 3. fuse and take top-k
    fused = _rrf_fuse([sem_ids, bm25_ids])[:k]

    hits = []
    for _id in fused:
        rec = records.get(_id)
        if rec is None:
            continue
        in_sem = _id in sem_dist
        in_bm25 = _id in bm25_ids
        hits.append({
            "id": _id,
            "document": rec["document"],
            "metadata": rec["metadata"],
            "distance": sem_dist.get(_id),  # None if BM25-only
            "source": "both" if (in_sem and in_bm25) else ("semantic" if in_sem else "keyword"),
        })
    # expose the best semantic distance so the caller can run the relevance gate
    best_dist = min(sem_dist.values()) if sem_dist else None
    return hits, best_dist


# -------------------------------------------------------------- generation ---
def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        price = f"Rs{m['price']:.0f}" if m.get("price", -1) >= 0 else "N/A"
        rating = f"{m['avg_rating']:.1f}/5" if m.get("avg_rating", -1) >= 0 else "unrated"
        blocks.append(
            f"[Product {i}] {m['name']} | Brand: {m['brand']} | Colour: {m['colour']} "
            f"| Price: {price} | Rating: {rating}\n{h['document']}"
        )
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    imgs = _img_map()
    out = []
    for h in hits:
        m = h["metadata"]
        out.append({
            "id": h["id"],
            "name": m["name"], "brand": m["brand"], "price": m["price"],
            "colour": m["colour"], "avg_rating": m.get("avg_rating", -1),
            "img": imgs.get(h["id"]),
            "distance": round(h["distance"], 3) if h.get("distance") is not None else None,
            "source": h.get("source", "semantic"),
        })
    return out


def _row_source(row) -> dict:
    imgs = _img_map()
    rid = str(int(row["product_id"]))
    return {
        "id": rid, "name": row["name"], "brand": row["brand"],
        "price": float(row["price"]) if pd.notna(row["price"]) else -1.0,
        "colour": row["colour"],
        "avg_rating": float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else -1.0,
        "img": imgs.get(rid), "distance": None, "source": "filter",
    }


# ---------------------------------------------------- filtered retrieval ----
def _semantic_raw(query: str, n: int) -> list[dict]:
    res = _collection().query(query_embeddings=[embed_query(query)], n_results=n)
    return [{"id": i, "document": d, "metadata": m, "distance": dist}
            for d, m, dist, i in zip(res["documents"][0], res["metadatas"][0],
                                     res["distances"][0], res["ids"][0])]


def _filtered_hybrid(query: str, k: int, allowed_ids: set[str]) -> list[dict]:
    """Hybrid retrieval restricted to products that passed the structured filter."""
    bm25, ids, records = _bm25_index()
    pool = min(max(200, k * 10), _collection().count())

    sem = [h for h in _semantic_raw(query, pool) if h["id"] in allowed_ids]
    sem_ids = [h["id"] for h in sem]
    sem_dist = {h["id"]: h["distance"] for h in sem}

    scores = bm25.get_scores(_tokenize(query))
    bm_ids = []
    for i in sorted(range(len(ids)), key=lambda j: scores[j], reverse=True):
        if scores[i] <= 0:
            break
        if ids[i] in allowed_ids:
            bm_ids.append(ids[i])
        if len(bm_ids) >= CANDIDATE_POOL:
            break

    fused = _rrf_fuse([sem_ids, bm_ids])[:k]
    # if neither retriever surfaced anything (e.g. tiny filtered set), just show them
    if not fused:
        fused = list(allowed_ids)[:k]
    hits = []
    for _id in fused:
        rec = records.get(_id)
        if not rec:
            continue
        in_sem, in_bm = _id in sem_dist, _id in bm_ids
        hits.append({"id": _id, "document": rec["document"], "metadata": rec["metadata"],
                     "distance": sem_dist.get(_id),
                     "source": "both" if (in_sem and in_bm) else ("semantic" if in_sem else "keyword")})
    return hits


# ----------------------------------------------------------- answer paths ----
def _empty(pq: dict, extra: str = "") -> dict:
    tail = extra or "Try relaxing one of the filters (colour, price, fabric or type)."
    return {"answer": f"I couldn't find anything in the catalogue matching **{qu.describe(pq)}**. {tail}",
            "sources": [], "used_fallback": True, "retrieved": [], "intent": pq["intent"]}


def _generate(query: str, hits: list[dict], intent: str = "search",
              filtered: bool = False) -> dict:
    context = _build_context(hits)
    if filtered:
        system = FILTERED_SYSTEM_PROMPT
        user_msg = (f"Pre-filtered matching products:\n{context}\n\n"
                    f"User question: {query}\n\n"
                    "Present these products (they already match the required filters).")
    else:
        system = SYSTEM_PROMPT
        user_msg = (f"Product context:\n{context}\n\nUser question: {query}\n\n"
                    "Answer using only the context above. If it is insufficient, say so.")
    resp = _openai().chat.completions.create(
        model=config.CHAT_MODEL, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_msg}])
    return {"answer": resp.choices[0].message.content.strip(),
            "sources": _sources(hits), "used_fallback": False,
            "retrieved": hits, "intent": intent}


def _search_path(query: str, pq: dict, k: int, mentions_men: bool) -> dict:
    has_filters = bool(pq["type"] or pq["colour_include"] or pq["colour_exclude"]
                       or pq["price_min"] is not None or pq["price_max"] is not None
                       or pq["fabrics"])
    if not has_filters:
        # no structured constraints -> original relevance-gated hybrid retrieval
        hits, best = retrieve_hybrid(query, k=k)
        if not hits or best is None or best > RELEVANCE_MAX_DISTANCE:
            return {"answer": FALLBACK_MESSAGE, "sources": [], "used_fallback": True,
                    "retrieved": [], "intent": "search"}
        return _generate(query, hits)

    allowed = set(structured_filter(pq, mentions_men)["id"])
    if not allowed:
        return _empty(pq)
    hits = _filtered_hybrid(query, k, allowed)
    if not hits:
        return _empty(pq)
    return _generate(query, hits, filtered=True)


def _count_path(pq: dict, mentions_men: bool) -> dict:
    df = structured_filter(pq, mentions_men)
    if df.empty:
        return _empty(pq)
    top = df.sort_values("avg_rating", ascending=False, na_position="last").head(6)
    return {"answer": f"There are **{len(df)}** products matching **{qu.describe(pq)}** in the catalogue.",
            "sources": [_row_source(r) for _, r in top.iterrows()],
            "used_fallback": False, "retrieved": [], "intent": "count"}


def _rank_path(pq: dict, intent: str, k: int, mentions_men: bool) -> dict:
    df = structured_filter(pq, mentions_men)
    if df.empty:
        return _empty(pq)
    if intent == "best":
        df = df[df["avg_rating"].notna()].sort_values(["avg_rating", "rating_count"], ascending=False)
        if df.empty:
            return _empty(pq, "None of the matching products have ratings yet.")
        label = "best rated"
    elif intent == "cheapest":
        df = df[df["price"] >= 0].sort_values("price", ascending=True)
        label = "cheapest"
    else:  # expensive
        df = df[df["price"] >= 0].sort_values("price", ascending=False)
        label = "most expensive"
    top = df.head(k)
    lines = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        rt = f"{r['avg_rating']:.1f}/5" if pd.notna(r["avg_rating"]) else "unrated"
        lines.append(f"{i}. **{r['name']}** — {r['brand']}, ₹{r['price']:.0f}, {rt}")
    head = f"The {label} {pq['type'] or 'products'} matching **{qu.describe(pq)}**:"
    return {"answer": head + "\n\n" + "\n".join(lines),
            "sources": [_row_source(r) for _, r in top.iterrows()],
            "used_fallback": False, "retrieved": [], "intent": intent}


def _fabric_of(_id: str) -> str:
    row = _catalog().set_index("id").loc[_id]
    m = re.search(r"Fabric[^:]*:\s*([^.]+)", str(row["attributes"]))
    return m.group(1).strip() if m else "—"


def _compare_path(query: str, pq: dict, k: int) -> dict:
    m = re.search(r"compare\s+(.+?)\s+(?:with|and|vs\.?|versus|to)\s+(.+)", query, re.I)
    if m:
        sides = [m.group(1), m.group(2)]
    else:
        parts = re.split(r"\s+vs\.?\s+|\s+versus\s+", query, flags=re.I)
        sides = parts if len(parts) == 2 else None
    if not sides:
        return _search_path(query, pq, k, False)

    found = [(_semantic_raw(s.strip(), 1) or [None])[0] for s in sides]
    if not all(found):
        missing = [s.strip() for s, f in zip(sides, found) if not f]
        return {"answer": f"I couldn't find a catalogue match for: {', '.join(missing)}.",
                "sources": [], "used_fallback": True, "retrieved": [], "intent": "compare"}

    a, b = found[0]["metadata"], found[1]["metadata"]
    def rt(m): return f"{m['avg_rating']:.1f}/5" if m.get("avg_rating", -1) >= 0 else "unrated"
    rows = [
        ("Colour", a["colour"], b["colour"]),
        ("Price", f"₹{a['price']:.0f}", f"₹{b['price']:.0f}"),
        ("Fabric", _fabric_of(found[0]["id"]), _fabric_of(found[1]["id"])),
        ("Rating", rt(a), rt(b)),
    ]
    tbl = f"| | {a['name'][:38]} | {b['name'][:38]} |\n|---|---|---|\n"
    for lbl, x, y in rows:
        tbl += f"| **{lbl}** | {x} | {y} |\n"
    return {"answer": tbl, "sources": _sources(found),
            "used_fallback": False, "retrieved": found, "intent": "compare"}


def answer(query: str, k: int = config.TOP_K, hybrid: bool = True,
           price_max: float | None = None, colour: str | None = None) -> dict:
    """Parse the query, route by intent, enforce hard filters, never hallucinate.

    `hybrid` is kept for API compatibility; filtered retrieval is always hybrid.
    `price_max` / `colour` are UI-sidebar overrides.
    """
    pq = qu.parse_query(query)
    if price_max is not None:
        pq["price_max"] = price_max
    if colour:
        pq["colour_include"] = {colour}
    mentions_men = (bool(re.search(r"\bmen'?s?\b|\bmale\b|\bgents\b", query.lower()))
                    and "women" not in query.lower())

    intent = pq["intent"]
    if intent == "compare":
        return _compare_path(query, pq, k)
    if intent == "count":
        return _count_path(pq, mentions_men)
    if intent in ("best", "cheapest", "expensive"):
        return _rank_path(pq, intent, k, mentions_men)
    return _search_path(query, pq, k, mentions_men)
