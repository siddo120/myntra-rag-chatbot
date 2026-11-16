"""Run the labelled benchmark against the live index.

    python eval/retrieval_eval.py

Modes: semantic and hybrid, each with and without the parsed query filters,
so the contribution of each stage is visible.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import build_queries, metrics  # noqa: E402
import index  # noqa: E402
import query as Q  # noqa: E402
import rag  # noqa: E402

st = index.load()
queries = build_queries(st.products.reset_index(drop=True))
rows = []
for group, q, rel in queries:
    for mode, hybrid, filters in [("semantic, no filters", False, False), ("hybrid, no filters", True, False),
                                  ("semantic + filters", False, True), ("hybrid + filters", True, True)]:
        spec = Q.parse(q, st.colours, st.brands) if filters else Q.Spec(text=q)
        hits, _, _ = rag.search(spec, 10, hybrid)
        rows.append({"group": group, "query": q, "n_rel": len(rel), "mode": mode,
                     **metrics([h["id"] for h in hits], rel),
                     "top3": [h["metadata"]["name"][:50] for h in hits[:3]]})
R = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "results_retrieval.json"
R.to_json(out, orient="records", indent=1)
pd.set_option("display.width", 220)
order = ["semantic, no filters", "hybrid, no filters", "semantic + filters", "hybrid + filters"]
print(R.pivot_table(index=["group", "query", "n_rel"], columns="mode", values="p5")[order].round(2).to_string())
print("\n== P@5 by group"); print(R.pivot_table(index="group", columns="mode", values="p5")[order].round(3).to_string())
print("\n== overall"); print(R.groupby("mode")[["p5", "mrr", "ndcg", "hit5"]].mean().round(3).loc[order].to_string())
