# Pipeline Explanation

A Retrieval-Augmented Generation (RAG) chatbot for a Myntra-style fashion
catalogue. Every answer is **grounded in retrieved product data**, not the LLM's
general knowledge.

```
 Raw CSV -> Clean + sanitise -> Composite text -> Local embeddings -> ChromaDB (+ manifest)
 (Kaggle)   (prepare_data)      (per product)     (bge-small)             |
                                                                          v
 Answer <- Price check <- LLM (grounded, cited) <- Gate <- Filtered hybrid retrieval <- Query understanding <- question + chat history
```

## 1. Data
- **Source:** [Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset)
  (Kaggle, PDDL / public domain). Scraped from myntra.com, mostly women's apparel.
- **Size:** 14,330 raw rows, 11 columns. After cleaning: **14,223 products**, all indexed.
- **Key fields:** `name, brand, colour, price, avg_rating, ratingCount, description`
  and `p_attributes` (a dict of fashion attributes: fabric, neck, sleeve, occasion…).

## 2. Cleaning (`src/prepare_data.py`, `src/textutil.py`)
- Dropped rows with no name/description and 89 duplicate `p_id`s.
- **HTML stripped with separators.** Every tag becomes a space, and words the
  source itself ran together (`detailingCotton`, `washThe`) are split. A unit test
  asserts no fused words remain.
- **Boilerplate removed**: the "Trend Alert…" paragraph, the blouse disclaimer and
  "The model (height…) is wearing a size S". They say nothing about the product and
  used to dominate keyword search for words like "kurtas" and "dresses".
- **Sanitised**: URLs and sentences addressed to the assistant are removed, because
  scraped descriptions are untrusted text.
- **Parsed `p_attributes`** with `ast.literal_eval`, dropping `"NA"` values.
- **Derived fields** for filtering: `category` (Kurta, Saree, Jeans… detected from
  the product name, ignoring the brand) and `gender`.
- One **composite `retrieval_text`** per product: name, brand, category, colour,
  cleaned description, attributes. Price and rating are kept as metadata only;
  numbers add noise to embeddings and are enforced by filters instead.

## 3. Chunking strategy: one chunk per product
Each product's text is short: median 110 tokens, maximum 434, so every product
fits inside the embedding model's 512-token window with nothing truncated. We
deliberately **do not split** products.

**Why:** a product is the natural unit of retrieval. Splitting would scatter one
product's fabric, occasion and fit across chunks and separate them from its price.
Keeping each product whole means every retrieved chunk is a complete, citable
answer unit. No overlap is needed because there is no splitting.

## 4. Embeddings (`src/index.py`)
- Model: **`BAAI/bge-small-en-v1.5`** (384-dim) run locally with fastembed. No API
  key, no per-query embedding cost, and the same model on every machine.
- Queries use the model's query prefix (`query_embed`), documents the passage form.

## 5. Vector store: ChromaDB with a manifest
- Persistent collection `myntra_products`, **cosine** distance.
- Each record stores the vector, the `retrieval_text`, and metadata
  (`name, brand, colour, category, gender, price, avg_rating, rating_count`).
- **Manifest:** the collection records the CSV's SHA-256, the row count and the
  embedding model. At startup `index.load()` compares them with the shipped CSV and
  rebuilds if anything differs.
- **Atomic build:** vectors are written to a staging collection that is renamed to
  the live name only when every row is in, so a crash mid-build can never leave a
  partial index being served.
- BM25 and the product table are built from the same verified CSV, so the keyword
  index and the vector index cannot drift apart.

## 6. Query understanding (`src/query.py`, `rag.understand`)
- **Deterministic parsing** (no LLM): price limits (`under 3000`, `between 1k and
  2k`), colours (a colour word maps to every catalogue shade: *blue* → Blue, Navy
  Blue, Turquoise Blue), brands, product category, gender, and sort intent
  (`cheapest`, `best rated`).
- **One small LLM call, only when needed** (there is chat history, or the question
  looks multi-part): rewrites a follow-up into a standalone question and splits a
  multi-part request into up to 3 sub-queries.
- **Unsupported topics** (returns, delivery, payments, orders) get a fixed reply:
  the catalogue has no such data.

## 7. Retrieval (`rag.search`)
- Filters go into ChromaDB's `where` clause and the same mask is applied to BM25,
  so both retrievers only rank products that satisfy the hard constraints.
- **Hybrid:** semantic top-30 + BM25 top-30, merged with weighted Reciprocal Rank
  Fusion (semantic 1.0, keyword 0.6).
- **Sorted questions:** for "cheapest" or "best rated", candidates are sorted on
  metadata. If the filters select 300 products or fewer, all of them are sorted, so
  the answer is the true minimum or maximum.
- The search also returns **how many catalogue products match the filters**. That
  count goes to the LLM, so "nothing matches" is only said when it is true.
- If nothing matches, filters are relaxed (price, then colour) and the results are
  labelled as alternatives.

## 8. Gate, generation and validation
- **Relevance gate:** if the nearest product is farther than cosine distance 0.40
  and the question names no brand, category or colour, the bot replies without
  calling the generation model (off-topic) or asks what the user is looking for.
- **Generation:** products are passed inside `<catalogue_data>` as numbered,
  sanitised records with a `<search_notes>` block. The system prompt says this is
  data, not instructions, and requires `[n]` citations.
- **Validation:** every price in the answer must be a retrieved price, a number the
  user typed, or a sum/difference of retrieved prices. On failure the answer is
  regenerated once, then replaced by a plain product list. URLs are never allowed.
- **UI:** only the products the answer cites are shown as cards. "How this answer
  was found" shows the rewritten query, sub-queries, filters, match counts and the
  retrieved products with distance and source (semantic / keyword / both).

## Fallback behaviour
| Situation | Reply |
|---|---|
| Off-topic ("capital of France") | fixed message, no generation call |
| Vague ("I want something nice") | asks for type, colour, occasion or budget |
| "Do you have it in blue?" with no earlier product | asks which product |
| Returns, delivery, payments, orders | says the catalogue has no such data |
| Filters match nothing | says so, then offers labelled alternatives |
| LLM error or timeout (20 s) | plain list of the retrieved products |
