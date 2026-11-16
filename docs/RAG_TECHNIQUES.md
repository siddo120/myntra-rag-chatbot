# RAG Techniques Used

## Baseline RAG (mandatory) ✅
Embed the question → cosine search in ChromaDB → grounded answer from the LLM with
an "insufficient context" fallback. See [PIPELINE.md](PIPELINE.md).

## Enhancement 1: Hybrid RAG ✅
Semantic retrieval and BM25 keyword retrieval, merged with **weighted Reciprocal
Rank Fusion** (`rag.search`, `rag.rrf_fuse`):

`score(doc) = Σ weight_r / (60 + rank_r(doc))`, semantic weight 1.0, keyword 0.6.

- **Semantic** matches meaning ("something warm with a hood" → hooded sweatshirts).
- **BM25** matches exact words: fabrics and techniques such as *chanderi*,
  *chikankari*, *gotta patti*. The tokeniser stems plurals and drops stopwords, and
  the product name is weighted double.
- Both retrievers run **inside the same metadata filters**, so the fusion only ever
  ranks products that satisfy the user's hard constraints.

The Streamlit toggle switches hybrid off to show the difference live.

## Enhancement 2: Query decomposition ✅
For a multi-part question, one small LLM call returns self-contained sub-queries
(`rag.understand`). Each sub-query is parsed, filtered and retrieved on its own,
and the evidence is presented to the answer model in labelled groups.

> "I need a red festive kurta under 3000 and a matching dupatta"
> → `red festive kurta under 3000` (category Kurta, colour Red, price ≤ 3000)
> → `red dupatta` (category Dupatta, colour Red)

The same call resolves follow-ups ("which of those is the cheapest?") into a
standalone question using the chat history. It is skipped for simple first
questions, so most turns cost one LLM call.

## Supporting techniques
- **Self-querying filters**: price, colour family, brand, category and gender are
  extracted from the text and applied as metadata filters (deterministic code, unit
  tested, no LLM).
- **Metadata aggregation**: "cheapest" and "best rated" sort on metadata rather
  than relying on top-k similarity.
- **Grounding checks**: numbered citations, a price validator, and evidence passed
  as untrusted data.

## Not implemented (future work)
- A cross-encoder reranker. The benchmark does not yet show a need for one.
- A policy/FAQ corpus for returns and delivery questions.
