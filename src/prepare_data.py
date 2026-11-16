"""
Clean and prepare the Myntra Fashion dataset for RAG.

Input : data/raw/Fashion Dataset.csv  (14,330 rows, HTML-laden descriptions)
Output: data/processed/products_clean.csv

Steps
-----
1. Drop rows with no name/description and deduplicate on p_id.
2. Strip HTML (tags become separators), drop stock boilerplate, and sanitise:
   URLs and instruction-like sentences are removed because scraped text is untrusted.
3. Parse the stringified p_attributes dict and pull out useful fashion attributes.
4. Derive filterable fields: category (from the name) and gender.
5. Build one composite 'retrieval_text' per product (what we embed). Price and
   rating stay out of it: they are metadata, and numbers add noise to embeddings.
"""

import ast

import pandas as pd

import config
from textutil import category_of, clean_html, drop_boilerplate, gender_of, sanitize

# p_attributes keys worth surfacing for retrieval (skip noisy/ID-like ones)
ATTR_KEYS = [
    "Fabric", "Top Fabric", "Bottom Fabric", "Dupatta Fabric",
    "Occasion", "Main Trend", "Print or Pattern Type", "Pattern",
    "Neck", "Sleeve Length", "Sleeve Styling",
    "Length", "Bottom Type", "Dupatta", "Shape", "Fit", "Waist Rise",
    "Weave Type", "Ornamentation", "Hemline", "Closure",
]


def parse_attrs(raw) -> dict:
    """p_attributes is a stringified python dict; parse safely, drop 'NA'."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items()
            if isinstance(v, str) and v.strip() and v.strip().upper() != "NA"}


def attrs_to_text(attrs: dict) -> str:
    return ". ".join(f"{key}: {attrs[key]}" for key in ATTR_KEYS if key in attrs)


def build_retrieval_text(row) -> str:
    segments = [f"{row['name']} by {row['brand']}. Category: {row['category']}. Colour: {row['colour']}."]
    if row["description_clean"]:
        segments.append(row["description_clean"])
    if row["attr_text"]:
        segments.append(row["attr_text"])
    return " ".join(segments)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["name", "description", "p_id"])
    df = df.drop_duplicates(subset=["p_id"], keep="first").copy()

    df["name"] = df["name"].map(lambda s: sanitize(clean_html(s, split_fused=False)))
    df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip()
    df["colour"] = df["colour"].fillna("Unknown").astype(str).str.strip()
    df["description_clean"] = df["description"].map(lambda s: sanitize(drop_boilerplate(clean_html(s))))
    df["attr_text"] = df["p_attributes"].map(parse_attrs).map(attrs_to_text).map(sanitize)
    # Detect the garment from the name without the brand ("MANY FROCKS & ..." is not a frock).
    df["category"] = [category_of(n[len(b):] if n.lower().startswith(b.lower()) else n)
                      for n, b in zip(df["name"], df["brand"])]
    df["gender"] = df["name"].map(gender_of)
    df["retrieval_text"] = df.apply(build_retrieval_text, axis=1)

    out = pd.DataFrame({
        "product_id": df["p_id"].astype("int64"),
        "name": df["name"],
        "brand": df["brand"],
        "colour": df["colour"],
        "category": df["category"],
        "gender": df["gender"],
        "price": df["price"],
        "avg_rating": df["avg_rating"],
        "rating_count": df["ratingCount"],
        "img": df["img"],
        "description": df["description_clean"],
        "attributes": df["attr_text"],
        "retrieval_text": df["retrieval_text"],
    })
    out = out[out["name"].str.len() > 0]
    out = out.dropna(subset=["price"])
    return out.sort_values("product_id").reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(config.RAW_CSV)
    print(f"raw rows: {len(raw)}")
    out = prepare(raw)
    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PROCESSED_CSV, index=False)

    rl = out["retrieval_text"].str.len()
    print(f"saved {len(out)} products -> {config.PROCESSED_CSV}")
    print(f"retrieval_text length: min {rl.min()}, median {int(rl.median())}, max {rl.max()}")
    print(f"rated: {out['avg_rating'].notna().mean()*100:.0f}% | brands: {out['brand'].nunique()}")
    print("categories:", out["category"].value_counts().to_dict())
    print("gender:", out["gender"].replace("", "(unlabelled)").value_counts().to_dict())


if __name__ == "__main__":
    main()
