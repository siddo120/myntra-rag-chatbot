"""Text helpers shared by ingestion and retrieval: HTML cleaning, sanitising
untrusted catalogue text, tokenising for BM25, and catalogue vocabularies."""

import html
import re

# ------------------------------------------------------------ HTML cleaning ---
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_LI_RE = re.compile(r"<li[^>]*>", re.I)

# Stock paragraphs that say nothing about the product but pollute both retrievers.
_BOILERPLATE = [
    re.compile(r"Trend Alert\b.*?(?:and more\.|$)", re.I | re.S),
    re.compile(r"The blouse worn by the model might be for modelling purpose only\.?", re.I),
    re.compile(r"Check the image of the blouse piece to understand how the actual blouse piece looks like\.?", re.I),
    re.compile(r"The model \(height[^)]*\) is wearing (?:a |an )?(?:free ?size|one ?size|size [\w.-]+)", re.I),
]


# The scraped source itself has fields run together ("detailingCotton", "washThe").
_FUSED_RE = re.compile(r"(?<=[a-z)])(?=[A-Z][a-z])")


def clean_html(text, split_fused: bool = True) -> str:
    """Strip tags/entities. Every tag becomes a separator so words never fuse."""
    if not isinstance(text, str):
        return ""
    text = _LI_RE.sub(" - ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text).replace("•", "-").replace("\xa0", " ")
    if split_fused:
        text = _FUSED_RE.sub(". ", text)
    return _WS_RE.sub(" ", text).strip()


def drop_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE:
        text = pat.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# -------------------------------------------------- untrusted-text sanitiser ---
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|in|net|org|io|co|example)\b\S*", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A sentence that addresses the assistant or tries to change its behaviour.
_INJECTION = re.compile(
    r"\b(?:assistant|chatbot|ai model|language model|system prompt|instructions?)\b"
    r"|\bignore (?:all |any |the )?(?:previous|prior|above)"
    r"|\b(?:tell|ask|instruct|inform) the (?:customer|user|shopper)"
    r"|\b(?:phone number|otp|password|card number|bank details)\b",
    re.I,
)


def sanitize(text: str) -> str:
    """Remove URLs and instruction-like sentences from catalogue text.

    Product descriptions are scraped, so they are untrusted. This runs at
    ingestion and again when the LLM context is built.
    """
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    kept = [s for s in _SENT_SPLIT.split(text) if not _INJECTION.search(s)]
    return _WS_RE.sub(" ", " ".join(kept)).strip()


# ------------------------------------------------------------ BM25 tokeniser ---
_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""
a an and are as at be by can do does for from has have i in is it its me my of on or our
that the their them this to was we were what which who will with you your show find want
need looking any some please there here also than then too very just about into over
""".split())


def stem(tok: str) -> str:
    """Light plural stemmer: enough to make 'kurtas' match 'kurta', 'dresses' match 'dress'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("sses", "shes", "ches", "xes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS]


# -------------------------------------------------------------- vocabularies ---
# Garment category, detected from the product name. First match by position wins,
# so "Kurta with Palazzos & Dupatta" is a Kurta and "Dress Material" is not a Dress.
CATEGORY_PATTERNS = [
    ("Dress Material", r"dress material|unstitched"),
    ("Lehenga", r"lehenga"),
    ("Saree", r"sarees?\b"),
    ("Kurta", r"kurtas?\b|kurtis?\b"),
    ("Dupatta", r"dupattas?\b|shawls?\b|stoles?\b"),
    ("Dress", r"dress(?:es)?\b|gowns?\b|frocks?\b"),
    ("Jumpsuit", r"jumpsuits?\b|playsuits?\b|dungarees?\b"),
    ("Co-ords", r"co-?ords?\b"),
    ("Jeans", r"jeans\b|jeggings\b"),
    ("Trousers", r"trousers?\b|palazzos?\b|pants?\b|leggings?\b|joggers?\b|culottes?\b|track pants"),
    ("Shorts", r"shorts\b"),
    ("Skirt", r"skirts?\b"),
    ("Sweatshirt", r"sweatshirts?\b|hoodies?\b"),
    ("Sweater", r"sweaters?\b|pullovers?\b|cardigans?\b"),
    ("Jacket", r"jackets?\b|blazers?\b|coats?\b|windcheater"),
    ("Shrug", r"shrugs?\b"),
    ("Top", r"tops?\b|tunics?\b|shirts?\b|t-?shirts?\b|tees?\b|blouses?\b"),
]
_CATEGORY_RES = [(name, re.compile(pat, re.I)) for name, pat in CATEGORY_PATTERNS]
# "top" is also an everyday word ("top rated", "top 5"); don't read those as the garment.
_NOT_GARMENT_TOP = re.compile(r"\btop(?=[\s-]*(?:rated|rating|\d|five|ten|three|picks?|sell|quality|brands?))", re.I)


_SAREE_BLOUSE = re.compile(r"saree\s+blouses?", re.I)


def find_categories(text: str) -> list[str]:
    """All categories mentioned in text, ordered by where they first appear."""
    text = _NOT_GARMENT_TOP.sub(" ", str(text))
    text = _SAREE_BLOUSE.sub("blouse", text)          # a "Saree Blouse" is a blouse, not a saree
    found = []
    for name, rx in _CATEGORY_RES:
        m = rx.search(text)
        if m:
            found.append((m.start(), name))
    # "Dress Material" also matches "dress": keep the more specific one.
    names = [n for _, n in sorted(found)]
    if "Dress Material" in names and "Dress" in names:
        names.remove("Dress")
    return names


def category_of(name: str) -> str:
    cats = find_categories(name)
    return cats[0] if cats else "Other"


_GENDER_RE = re.compile(r"\b(women|woman|ladies|men|mens|girls?|boys?|kids|unisex)\b", re.I)
_GENDER_MAP = {"women": "Women", "woman": "Women", "ladies": "Women", "men": "Men", "mens": "Men",
               "girl": "Girls", "girls": "Girls", "boy": "Boys", "boys": "Boys", "kids": "Kids",
               "unisex": "Unisex"}


def gender_of(text: str) -> str:
    """Gender/age group named in a product name or a query ('' if none)."""
    m = _GENDER_RE.search(str(text).replace("'s", "s").replace("’s", "s"))
    return _GENDER_MAP[m.group(1).lower()] if m else ""
