"""
Streamlit chat UI for the Myntra Fashion RAG chatbot.

Run:  streamlit run app.py
"""

import html
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# set_page_config MUST be the first Streamlit command on the page.
st.set_page_config(page_title="Myntra Fashion Assistant", page_icon="🛍️", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("myntra.app")

# On Streamlit Cloud the API key is provided via st.secrets; bridge it into the
# environment so src/config.py picks it up the same way as the local .env file.
try:
    if st.secrets.load_if_toml_exists():
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
            if name in st.secrets:
                os.environ[name] = str(st.secrets[name])
except Exception:
    pass  # malformed/absent secrets file -> fall back to .env locally

import config
import index
import rag


@st.cache_resource(show_spinner="Loading the product index (first run builds it, a few minutes)...")
def _warm_index() -> int:
    """Verify the shipped index against the catalogue; rebuild it if it does not match."""
    return len(index.load().products)


try:
    CATALOGUE_SIZE = _warm_index()
except Exception:
    log.exception("index failed to load")
    st.error("The product index could not be loaded. Please check the server logs.")
    st.stop()

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      .stApp { background: #fafafa; }
      h1 { color: #ff3f6c; }  /* Myntra pink */
      .product-card {
        border: 1px solid #ececec; border-radius: 12px; padding: 10px;
        background: #ffffff; height: 100%;
      }
      .product-card .pname { font-weight: 600; font-size: 0.82rem; line-height: 1.2;
        margin: 6px 0 2px; color: #282c3f; }
      .product-card .pmeta { font-size: 0.74rem; color: #7e818c; }
      .product-card .pprice { font-weight: 700; color: #282c3f; font-size: 0.9rem; }
      .badge { font-size: 0.65rem; padding: 1px 6px; border-radius: 8px;
        background: #fff0f3; color: #ff3f6c; border: 1px solid #ffd4de; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Suggest a black cotton kurta for a festive occasion",
    "Red festive kurta under 3000 and a matching dupatta",
    "Compare a Biba anarkali with a Libas kurta set",
    "What is the best rated saree under 2000?",
]

BADGE = {"both": "🔵🟢 both", "semantic": "🔵 semantic", "keyword": "🟢 keyword", "previous": "↩︎ earlier result"}


def render_products(sources: list[dict]) -> None:
    """Show the products the answer actually cited."""
    if not sources:
        return
    st.caption("🛍️ Products cited in this answer")
    cols = st.columns(min(len(sources), 5))
    for i, s in enumerate(sources[:10]):
        with cols[i % len(cols)]:
            rating = f"⭐ {s['avg_rating']:.1f}" if s.get("avg_rating", -1) >= 0 else ""
            if isinstance(s.get("img"), str) and s["img"].startswith("http"):
                st.image(s["img"], use_container_width=True)
            # Catalogue text is untrusted: escape it before it goes into HTML.
            st.markdown(
                f"<div class='product-card'>"
                f"<div class='pname'>[{s['n']}] {html.escape(s['name'][:60])}</div>"
                f"<div class='pmeta'>{html.escape(s['brand'])} · {html.escape(s['colour'])}</div>"
                f"<div class='pprice'>₹{s['price']:.0f} <span class='pmeta'>{rating}</span></div>"
                f"<span class='badge'>{BADGE.get(s.get('source'), '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_trace(trace: dict) -> None:
    """Show how the answer was produced: rewritten query, filters, retrieved chunks."""
    if not trace:
        return
    with st.expander("How this answer was found"):
        plan = trace.get("plan") or {}
        if plan.get("rewritten"):
            st.markdown(f"**Understood as:** {plan['standalone']}")
            if len(plan.get("sub_queries", [])) > 1:
                st.markdown("**Split into:** " + " · ".join(f"`{s}`" for s in plan["sub_queries"]))
        for note in plan.get("notes", []):
            st.markdown(note)
        rows = [{"#": i, "product": r["name"], "price": r["price"], "found by": r["source"],
                 "distance": r["distance"]} for i, r in enumerate(trace.get("retrieved", []), 1)]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🛍️ Myntra Assistant")
    st.caption(f"RAG chatbot grounded in a Myntra fashion catalogue of {CATALOGUE_SIZE:,} products.")

    st.subheader("Search settings")
    use_hybrid = st.toggle("Hybrid search (semantic + keyword)", value=True,
                           help="Combine meaning-based vector search with BM25 "
                                "keyword matching. Better at exact terms like "
                                "brand names and fabrics.")
    top_k = st.slider("Products to retrieve (top-k)", 3, 10, config.TOP_K)

    use_price = st.checkbox("Filter by max price")
    price_max = st.slider("Max price (₹)", 200, 20000, 3000, step=100) if use_price else None

    use_colour = st.checkbox("Filter by colour")
    colour = (
        st.selectbox("Colour", ["Black", "White", "Blue", "Red", "Green", "Pink",
                                 "Yellow", "Maroon", "Grey", "Beige",
                                 "Purple", "Orange", "Brown"],
                     help="Includes shades: Blue also matches Navy Blue and Turquoise Blue.")
        if use_colour else None
    )

    show_sources = st.checkbox("Show product images", value=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Pipeline: question → rewrite & filters → hybrid retrieval (ChromaDB + BM25) → "
               "grounded answer with citations → price check. Prices and budgets in your "
               "question are applied as filters automatically.")

# ---------------- Header ----------------
st.title("🛍️ Myntra Fashion Assistant")
st.write("Ask about products, fabrics, fit, price, colour or occasion — "
         "answers come **only** from the catalogue.")

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
example_clicked = None
for col, ex in zip(ex_cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        example_clicked = ex

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_products(msg["sources"])
        render_trace(msg.get("trace"))


def handle_query(prompt: str) -> None:
    history = [{"role": m["role"], "content": m["content"], "sources": m.get("sources")}
               for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the catalogue..."):
            try:
                result = rag.answer(prompt, history=history, k=top_k, hybrid=use_hybrid,
                                    price_max=price_max, colour=colour)
            except config.MissingKeyError as e:
                result = {"answer": f"⚠️ {e}", "sources": []}
            except Exception:
                # Details go to the server log, never to the user.
                log.exception("answer failed")
                result = {"answer": "⚠️ Something went wrong while answering. Please try again.",
                          "sources": []}
        st.markdown(result["answer"])
        sources = result.get("sources") or []
        trace = None
        if result.get("retrieved"):
            trace = {"plan": result.get("plan"),
                     "retrieved": [{"name": h["metadata"]["name"], "price": h["metadata"]["price"],
                                    "source": h["source"],
                                    "distance": None if h["distance"] is None else round(h["distance"], 3)}
                                   for h in result["retrieved"]]}
        if show_sources:
            render_products(sources)
        render_trace(trace)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"],
                                      "sources": sources, "trace": trace})


# ---------------- Input ----------------
typed = st.chat_input("Ask about Myntra fashion products...")
query = typed or example_clicked
if query:
    handle_query(query)
