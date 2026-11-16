# 🛍️ Myntra Fashion RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a
Myntra-style fashion catalogue — product discovery, fabric, fit, price, colour,
occasion — **grounded in retrieved product data**, with a Streamlit chat UI.

Built for the RAG Chatbot assignment: baseline RAG, plus **Hybrid RAG** (semantic +
BM25 keyword) and **query decomposition** for multi-part questions.

## What it does
- Answers fashion queries using **only** the catalogue (14,223 products), not the
  LLM's general knowledge, and cites the products it used as `[1]`, `[2]`.
- **Understands the question first**: budgets ("under ₹3000"), colours, brands,
  product types and "cheapest / best rated" are parsed into hard filters and sort
  orders, so they are enforced instead of guessed.
- **Hybrid retrieval**: semantic vector search (ChromaDB) fused with BM25 keyword
  search via weighted Reciprocal Rank Fusion.
- **Follow-up questions work**: "which of those is the cheapest?" is resolved
  against the conversation.
- **Multi-part questions are decomposed**: "a red kurta under 3000 and a matching
  dupatta" becomes two searches.
- **Guardrails**: off-topic and unsupported questions (returns, delivery) get a
  clear reply; catalogue text is treated as untrusted data; every price in an
  answer is checked against the retrieved products before it is shown.

## Tech stack
| Layer | Choice |
|-------|--------|
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally with fastembed (no API key) |
| LLM | Groq `openai/gpt-oss-120b` by default; OpenAI `gpt-4o-mini` if only that key is set |
| Vector store | ChromaDB (persistent, cosine), with metadata filters |
| Keyword search | `rank-bm25` (BM25Okapi) with plural stemming and stopwords |
| UI | Streamlit |
| Data | [Myntra Fashion Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) (Kaggle, PDDL) |

## Project structure
```
myntra-rag-chatbot/
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── src/
│   ├── config.py          # paths, models, thresholds, env loading
│   ├── textutil.py        # HTML cleaning, sanitising, tokeniser, category/gender vocab
│   ├── prepare_data.py    # raw CSV -> products_clean.csv
│   ├── index.py           # atomic index build, manifest, startup verification, BM25
│   ├── build_index.py     # CLI: embed products -> chroma_db/
│   ├── query.py           # deterministic query parsing: filters, sort, intents
│   └── rag.py             # understand -> retrieve -> gate -> generate -> validate
├── tests/
│   └── test_units.py      # offline unit tests (no key, no network)
├── eval/
│   ├── benchmark.py       # labelled retrieval queries + metrics
│   ├── retrieval_eval.py  # P@5 / MRR / nDCG per retrieval mode
│   ├── gate_analysis.py   # relevance-gate calibration
│   └── generation_suite.py# end-to-end stress cases (needs an LLM key)
├── docs/
│   ├── PIPELINE.md         # data -> answer explanation
│   ├── RAG_TECHNIQUES.md   # baseline, hybrid, decomposition
│   └── TESTING.md          # results + failure analysis
├── data/
│   ├── raw/                # Fashion Dataset.csv (downloaded, git-ignored)
│   └── processed/          # products_clean.csv (shipped)
└── chroma_db/              # prebuilt vector index for the full catalogue (shipped)
```

## Setup

### 1. Install
```bash
cd myntra-rag-chatbot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add an LLM key
Create `kaggle-rag-chatbot.env` (git-ignored) with one of:
```
GROQ_API_KEY=gsk_xxxxxxxx         # free tier works: console.groq.com
OPENAI_API_KEY=sk-xxxxxxxx        # alternative
```
Embeddings run locally, so the key is only used for answers.

### 3. Run the chatbot
```bash
streamlit run app.py
```
Open http://localhost:8501. The repo ships the cleaned catalogue and a prebuilt
index, so nothing needs to be built. At startup the app checks that the index
matches `products_clean.csv` (hash, row count, embedding model) and rebuilds it
if it does not.

### Rebuilding from the raw data (optional)
```bash
kaggle datasets download -d hiteshsuthar101/myntra-fashion-product-dataset \
  -f "Fashion Dataset.csv" -p data/raw && unzip -o data/raw/*.zip -d data/raw
python src/prepare_data.py     # clean -> data/processed/products_clean.csv
python src/build_index.py      # embed 14,223 products locally (~15 min on a laptop CPU)
```

## Testing
```bash
python -m pytest tests/test_units.py      # offline unit tests
python eval/retrieval_eval.py             # labelled retrieval benchmark
python eval/gate_analysis.py              # relevance-gate check
python eval/generation_suite.py           # end-to-end cases (uses the LLM key)
```
See [docs/TESTING.md](docs/TESTING.md) for results and documented failure cases.

## Demo tip (5-minute presentation)
1. Ask *"Red festive kurta under 3000 and a matching dupatta"*, then open **How this
   answer was found**: it shows the two sub-queries, the parsed filters and how many
   catalogue products match.
2. Follow up with *"Which of those is the cheapest?"* to show conversation memory.
3. Toggle **Hybrid search** off and on for a fabric question (*"chanderi dupatta"*) to
   show the keyword retriever's contribution.
4. Ask *"What is Myntra's return policy?"* or *"What is the capital of France?"* to
   show the fallbacks.

## Notes & limitations
- The catalogue is **women's fashion** (kurtas, sarees, tops, jeans, jackets). It has
  very few dresses and almost no men's wear; the bot says so rather than guessing.
- There is no stock, size-availability, delivery or returns data.
- "Best rated" ignores products with fewer than 10 ratings.
