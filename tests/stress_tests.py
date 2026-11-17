"""
Stress tests for the Myntra RAG chatbot.

Each case targets a known failure mode for naive RAG. Run:
    python tests/stress_tests.py

Prints the parsed intent, whether the fallback fired, the answer, and the
colours/prices of the products shown — so you can eyeball correctness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import query_understanding as qu  # noqa: E402
import rag  # noqa: E402

CASES = [
    ("Negation (colour)",      "suggest a kurta but not in black color"),
    ("Superlative + rating",   "what is the best rated saree under 2000?"),
    ("Price constraint",       "show me a lehenga under 1500"),
    ("Multi-constraint",       "red cotton festive kurta under 3000 with dupatta"),
    ("Type drift guard",       "silk saree between 1000 and 3000"),
    ("Scope: men's",           "do you have a men's formal shirt?"),
    ("Scope: daughter/kid",    "a dress for my daughter"),
    ("Compound colour",        "navy kurta"),
    ("Counting",               "how many red sarees do you have?"),
    ("Comparison",             "compare a Biba anarkali with a Libas kurta set"),
    ("Cheapest",               "cheapest silk saree"),
    ("Empty but plausible",    "neon green blazer under 500"),
    ("Out-of-domain",          "what is the capital of France?"),
    ("Double negation",        "kurta not black and not white"),
]


def run():
    for label, q in CASES:
        pq = qu.parse_query(q)
        out = rag.answer(q)
        print("=" * 100)
        print(f"[{label}]  Q: {q}")
        print(f"   intent={out.get('intent')}  fallback={out['used_fallback']}  parsed=[{qu.describe(pq)}]")
        print("   A:", out["answer"][:360].replace("\n", " "))
        srcs = out.get("sources") or []
        if srcs:
            shown = [f"{s['colour']}/₹{s['price']:.0f}" for s in srcs[:5]]
            print("   shown:", shown)


if __name__ == "__main__":
    run()
