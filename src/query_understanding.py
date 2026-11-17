"""
Query understanding for the Myntra RAG chatbot.

Turns a free-text query into STRUCTURED filters + an intent, so retrieval can
enforce hard constraints (colour incl. negation, price, fabric, type, occasion)
instead of relying on semantic similarity alone. This is what fixes:
  - negation ("kurta but not black")
  - price constraints ("under 3000")
  - superlatives ("best rated saree")  -> intent=best
  - counting ("how many red sarees")   -> intent=count
  - comparison ("compare A vs B")      -> intent=compare
"""

import re

# ---- canonical colours present in the catalogue (47) ----
CANON_COLOURS = [
    "Beige", "Black", "Blue", "Bronze", "Brown", "Burgundy", "Camel Brown",
    "Champagne", "Charcoal", "Coffee Brown", "Copper", "Coral", "Cream",
    "Fluorescent Green", "Fuchsia", "Gold", "Green", "Grey", "Grey Melange",
    "Khaki", "Lavender", "Lime Green", "Magenta", "Maroon", "Mauve", "Multi",
    "Mustard", "Navy Blue", "Nude", "Off White", "Olive", "Orange", "Peach",
    "Pink", "Purple", "Red", "Rose", "Rust", "Sea Green", "Silver", "Tan",
    "Taupe", "Teal", "Turquoise Blue", "Violet", "White", "Yellow",
]

# user word (lowercase) -> canonical colour value
COLOUR_SYNONYMS = {
    "navy": "Navy Blue", "offwhite": "Off White", "off-white": "Off White",
    "ivory": "Off White", "wine": "Burgundy", "turquoise": "Turquoise Blue",
    "gray": "Grey", "coffee": "Coffee Brown", "camel": "Camel Brown",
    "multicolour": "Multi", "multicolor": "Multi", "multi-colour": "Multi",
}

# broad family words -> set of catalogue colours
COLOUR_FAMILIES = {
    "red": {"Red", "Maroon", "Rust", "Burgundy", "Fuchsia", "Magenta", "Coral", "Rose"},
    "blue": {"Blue", "Navy Blue", "Turquoise Blue", "Teal", "Sea Green"},
    "green": {"Green", "Olive", "Lime Green", "Sea Green", "Teal", "Fluorescent Green"},
    "white": {"White", "Off White", "Cream"},
    "brown": {"Brown", "Camel Brown", "Coffee Brown", "Tan", "Taupe", "Khaki", "Beige"},
    "grey": {"Grey", "Grey Melange", "Charcoal"},
    "purple": {"Purple", "Violet", "Mauve", "Lavender"},
}

# product-type word -> regex matched against the product NAME (lowercased)
TYPE_PATTERNS = {
    "kurta": r"kurt",          # kurta or kurti
    "saree": r"saree|\bsari\b",
    "lehenga": r"lehenga|lehnga|lehanga",
    "dress": r"\bdress\b|gown|frock",
    "top": r"\btop\b|tunic|tshirt|t-shirt",
    "trouser": r"trouser|palazzo|pant",
    "dupatta": r"dupatta",
    "skirt": r"skirt",
    "jumpsuit": r"jumpsuit",
    "shrug": r"shrug",
    "jacket": r"jacket|blazer|coat",
    "shirt": r"\bshirt\b",
    "jeans": r"jeans|denim",
    "co-ord": r"co-ord|coord",
    "sharara": r"sharara|garara",
    "suit": r"salwar|\bsuit\b",
    "sweater": r"sweater|pullover|cardigan",
    "saree blouse": r"blouse",
}

# occasion word -> set of catalogue Occasion values
OCCASION_MAP = {
    "festive": {"Festive", "Ethnic", "Traditional", "Party"},
    "diwali": {"Festive", "Ethnic", "Traditional"},
    "deepavali": {"Festive", "Ethnic", "Traditional"},
    "navratri": {"Festive", "Ethnic", "Traditional"},
    "wedding": {"Festive", "Ethnic", "Traditional", "Party"},
    "sangeet": {"Festive", "Party"},
    "reception": {"Festive", "Party"},
    "puja": {"Ethnic", "Traditional", "Festive"},
    "pooja": {"Ethnic", "Traditional", "Festive"},
    "ethnic": {"Ethnic", "Traditional", "Festive"},
    "traditional": {"Traditional", "Ethnic"},
    "celebration": {"Festive", "Party"},
    "party": {"Party"},
    "cocktail": {"Party"},
    "casual": {"Casual", "Daily"},
    "daily": {"Casual", "Daily"},
    "everyday": {"Casual", "Daily"},
    "formal": {"Formal", "Work"},
    "office": {"Formal", "Work"},
    "work": {"Work", "Formal"},
    "sports": {"Sports"},
    "gym": {"Sports"},
    "workout": {"Sports"},
}

FABRICS = [
    "cotton", "silk", "rayon", "viscose", "georgette", "chiffon", "linen",
    "denim", "velvet", "chanderi", "organza", "wool", "woollen", "net",
    "satin", "crepe", "polyester", "nylon", "khadi", "modal", "lycra",
    "jute", "brocade", "tissue", "schiffli", "handloom",
]

_NEG_CUES = r"(?:not|no|without|except|avoid|other than|excluding|non)"
_NUM = r"(\d[\d,]*)\s*(k)?"


def _norm_num(num: str, k: str | None) -> float:
    val = float(num.replace(",", ""))
    if k:
        val *= 1000
    return val


def _colour_phrases():
    """All colour phrases (canonical + synonyms + families), longest first."""
    phrases = {c.lower(): ("colour", c) for c in CANON_COLOURS}
    for w, c in COLOUR_SYNONYMS.items():
        phrases[w] = ("colour", c)
    for fam in COLOUR_FAMILIES:
        phrases[fam] = ("family", fam)
    # longest phrase first so "navy blue" beats "blue"
    return sorted(phrases.items(), key=lambda kv: -len(kv[0]))


_COLOUR_PHRASES = _colour_phrases()


def _expand_colour(kind: str, val: str) -> set[str]:
    return set(COLOUR_FAMILIES[val]) if kind == "family" else {val}


def parse_colours(text: str) -> tuple[set[str], set[str]]:
    """Return (include, exclude) sets of canonical colours."""
    include, exclude = set(), set()
    low = text.lower()
    used_spans = []
    for phrase, (kind, val) in _COLOUR_PHRASES:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", low):
            span = m.span()
            if any(s[0] <= span[0] < s[1] for s in used_spans):
                continue  # already consumed by a longer phrase
            used_spans.append(span)
            # negation if a negation cue appears shortly before the colour
            pre = low[max(0, span[0] - 18):span[0]]
            target = exclude if re.search(_NEG_CUES + r"[\sa-z]*$", pre) else include
            target |= _expand_colour(kind, val)
    # a colour can't be both; exclusion wins over an accidental include
    include -= exclude
    return include, exclude


def parse_price(text: str) -> tuple[float | None, float | None]:
    low = text.lower()
    pmin = pmax = None
    m = re.search(rf"between\s+{_NUM}\s+(?:and|to|-)\s+{_NUM}", low)
    if m:
        a = _norm_num(m.group(1), m.group(2))
        b = _norm_num(m.group(3), m.group(4))
        return min(a, b), max(a, b)
    m = re.search(rf"(?:under|below|less than|upto|up to|within|cheaper than|max|<=?)\s*(?:rs\.?|₹|inr)?\s*{_NUM}", low)
    if m:
        pmax = _norm_num(m.group(1), m.group(2))
    m = re.search(rf"(?:over|above|more than|minimum|at least|>=?)\s*(?:rs\.?|₹|inr)?\s*{_NUM}", low)
    if m:
        pmin = _norm_num(m.group(1), m.group(2))
    m = re.search(rf"around\s*(?:rs\.?|₹)?\s*{_NUM}", low)
    if m and pmax is None and pmin is None:
        v = _norm_num(m.group(1), m.group(2))
        pmin, pmax = 0.8 * v, 1.2 * v
    return pmin, pmax


def parse_type(text: str) -> str | None:
    low = text.lower()
    for canon, pat in TYPE_PATTERNS.items():
        if re.search(pat, low):
            return canon
    return None


def parse_occasions(text: str) -> set[str]:
    low = text.lower()
    occ = set()
    for word, vals in OCCASION_MAP.items():
        if re.search(rf"\b{word}\b", low):
            occ |= vals
    return occ


def parse_fabrics(text: str) -> set[str]:
    low = text.lower()
    return {f for f in FABRICS if re.search(rf"\b{f}\b", low)}


def parse_intent(text: str) -> str:
    low = text.lower()
    if re.search(r"\bcompare\b|\bvs\.?\b|\bversus\b|difference between", low):
        return "compare"
    if re.search(r"how many|number of|count of|how much choice", low):
        return "count"
    if re.search(r"cheapest|lowest price|least expensive|most affordable|budget", low):
        return "cheapest"
    if re.search(r"most expensive|costliest|highest price|priciest", low):
        return "expensive"
    if re.search(r"best rated|top rated|highest rated|best.*rating|most popular|best seller|bestseller", low):
        return "best"
    return "search"


def parse_query(text: str) -> dict:
    inc, exc = parse_colours(text)
    pmin, pmax = parse_price(text)
    return {
        "raw": text,
        "intent": parse_intent(text),
        "colour_include": inc,
        "colour_exclude": exc,
        "price_min": pmin,
        "price_max": pmax,
        "type": parse_type(text),
        "occasions": parse_occasions(text),
        "fabrics": parse_fabrics(text),
        "with_dupatta": bool(re.search(r"\bdupatta\b", text.lower())),
    }


def describe(pq: dict) -> str:
    """Human-readable summary of the parsed constraints (for fallbacks/debug)."""
    bits = []
    if pq["type"]:
        bits.append(pq["type"])
    if pq["colour_include"]:
        bits.append("colour " + "/".join(sorted(pq["colour_include"])))
    if pq["colour_exclude"]:
        bits.append("not " + "/".join(sorted(pq["colour_exclude"])))
    if pq["fabrics"]:
        bits.append("/".join(sorted(pq["fabrics"])))
    if pq["occasions"]:
        bits.append("for " + "/".join(sorted(pq["occasions"])))
    if pq["price_max"]:
        bits.append(f"under Rs{int(pq['price_max'])}")
    if pq["price_min"]:
        bits.append(f"over Rs{int(pq['price_min'])}")
    return ", ".join(bits) if bits else "your query"
