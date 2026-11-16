"""Labelled retrieval benchmark.

A product is relevant to a query iff it satisfies the query's predicate over
catalogue fields, so labels are reproducible and need no hand judging.
"""
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from textutil import category_of  # noqa: E402


def build_queries(df):
    """Return [(group, query, set_of_relevant_ids)] for a products dataframe."""
    low = lambda s: str(s).lower()
    n = df["name"].map(low)
    t = (df["name"].astype(str) + " " + df["description"].fillna("").astype(str) + " "
         + df["attributes"].fillna("").astype(str)).map(low)
    brand_less = [nm[len(b):] if low(nm).startswith(low(b)) else nm for nm, b in zip(df["name"], df["brand"])]
    cat = df["category"] if "category" in df else pd.Series([category_of(x) for x in brand_less], index=df.index)

    def attr(k):
        return df["attributes"].fillna("").str.extract(re.escape(k) + r": ([^.]+)")[0].fillna("").str.lower()

    occ, orn, neck, pat = attr("Occasion"), attr("Ornamentation"), attr("Neck"), attr("Print or Pattern Type")
    fabric = attr("Fabric") + " " + attr("Top Fabric")
    is_ = lambda c: cat == c
    rated = (df["avg_rating"] >= 4.5) & (df["rating_count"].fillna(0) >= 10)
    Q = [
        ("lexical", "Do you have a Biba anarkali kurta?", (df.brand == "Biba") & t.str.contains("anarkali") & is_("Kurta")),
        ("lexical", "Dupatta Bazaar chanderi dupatta", (df.brand == "Dupatta Bazaar") & t.str.contains("chanderi")),
        ("lexical", "MANGO black jacket", (df.brand == "MANGO") & (df.colour == "Black") & is_("Jacket")),
        ("lexical", "Roadster blue jeans", (df.brand == "Roadster") & df.colour.str.contains("Blue") & is_("Jeans")),
        ("lexical", "dupatta with gotta patti work", is_("Dupatta") & orn.str.contains("gotta")),
        ("lexical", "chikankari kurta", is_("Kurta") & t.str.contains("chikankari")),
        ("lexical", "sequinned saree", is_("Saree") & (orn.str.contains("sequin") | n.str.contains("sequin"))),
        ("lexical", "mirror work lehenga", is_("Lehenga") & t.str.contains("mirror")),
        ("semantic", "something warm with a hood for winter", neck.str.contains("hood") | n.str.contains("hood")),
        ("semantic", "six-yard drape with a golden zari border", is_("Saree") & t.str.contains("zari")),
        ("semantic", "denim bottoms", is_("Jeans") | (n.str.contains("denim") & (is_("Shorts") | is_("Skirt") | is_("Trousers")))),
        ("semantic", "outfit to wear to a wedding sangeet", (is_("Lehenga") | is_("Saree") | is_("Kurta")) & occ.str.contains("festive|ethnic|traditional|party")),
        ("semantic", "smart clothes for the office", occ.str.contains("formal|work")),
        ("semantic", "a flowy floral summer frock", is_("Dress") & (pat.str.contains("floral") | n.str.contains("floral"))),
        ("morphology", "black cotton kurtas", is_("Kurta") & (df.colour == "Black") & fabric.str.contains("cotton")),
        ("morphology", "floral dresses", is_("Dress") & (pat.str.contains("floral") | n.str.contains("floral"))),
        ("morphology", "embroidered sarees", is_("Saree") & t.str.contains("embroider")),
        ("constraint", "red kurta under 1000 rupees", is_("Kurta") & (df.colour == "Red") & (df.price <= 1000)),
        ("constraint", "yellow saree below 1500", is_("Saree") & (df.colour == "Yellow") & (df.price <= 1500)),
        ("constraint", "jeans above 3000", is_("Jeans") & (df.price > 3000)),
        ("constraint", "highest rated lehenga", is_("Lehenga") & rated),
    ]
    ids = df["product_id"].astype(int).astype(str)
    return [(g, q, set(ids[m.to_numpy()])) for g, q, m in Q]


def metrics(ranked, rel):
    g = [1 if r in rel else 0 for r in ranked[:10]]
    p5 = sum(g[:5]) / 5
    mrr = next((1 / (i + 1) for i, x in enumerate(g) if x), 0.0)
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    return {"p5": p5, "mrr": mrr, "ndcg": dcg / idcg if idcg else 0.0,
            "hit5": float(any(g[:5]))}
