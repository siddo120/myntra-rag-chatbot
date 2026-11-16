"""Query understanding: turn free text into hard constraints, a sort order and
an intent, without an LLM. Everything here is deterministic and unit-tested."""

import re
from dataclasses import dataclass, field

from textutil import CATEGORY_PATTERNS, find_categories, gender_of, stem, tokenize

_CATEGORY_WORDS = re.compile("|".join(p for _, p in CATEGORY_PATTERNS), re.I)

# ------------------------------------------------------------------- prices ---
_NUM = r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?"
_MAX_RE = re.compile(r"(?:under|below|less than|cheaper than|within|upto|up to|at most|max(?:imum)?(?: price)?(?: of)?|budget(?: of| is)?|not more than|<=?)\s*" + _NUM, re.I)
_MIN_RE = re.compile(r"(?:above|over|more than|at least|min(?:imum)?(?: price)?(?: of)?|starting (?:at|from)|>=?)\s*" + _NUM, re.I)
_RANGE_RE = re.compile(r"(?:between|from)\s*" + _NUM + r"\s*(?:and|to|-)\s*" + _NUM, re.I)


def _amount(digits: str, k: str | None) -> float:
    return float(digits.replace(",", "")) * (1000 if k else 1)


# --------------------------------------------------------------------- sort ---
_SORTS = [
    ("price_asc", re.compile(r"\b(?:cheapest|least expensive|lowest[- ]priced?|lowest price|most affordable|most budget)\b", re.I)),
    ("price_desc", re.compile(r"\b(?:most expensive|priciest|costliest|highest[- ]priced?|highest price|most premium)\b", re.I)),
    ("rating_desc", re.compile(r"\b(?:best|highest|top)[- ](?:rated|rating|reviewed|reviews)\b|\bmost (?:popular|loved)\b|\bbest ratings?\b", re.I)),
]

# ------------------------------------------------------------------ intents ---
# Things shoppers ask that a product catalogue cannot answer.
_UNSUPPORTED = [
    ("returns, refunds or exchanges", re.compile(r"\b(?:return|refund|exchange|replacement)s?\b(?:.{0,20}\bpolic(?:y|ies)\b)?|\bpolic(?:y|ies)\b", re.I)),
    ("delivery or shipping", re.compile(r"\b(?:deliver(?:y|ed|ies)?|shipping|shipped|dispatch|courier)\b", re.I)),
    ("payments", re.compile(r"\b(?:cash on delivery|cod|emi|payment options?|pay (?:by|with|using))\b", re.I)),
    ("orders", re.compile(r"\b(?:track|cancel) (?:my |an |the )?order\b|\bmy order\b|\border status\b", re.I)),
]
_REFERENCE_RE = re.compile(r"\b(?:it|its|that one|this one|those|these|them|the (?:first|second|third|last) one|the same)\b", re.I)

# Colour words that are too often something else ("gold-toned zari", "rose print").
_AMBIGUOUS_COLOURS = {"gold", "silver", "copper", "bronze", "rose", "multi", "assorted", "champagne",
                      "nude", "tan", "melange"}
_COLOUR_ALIASES = {"gray": "grey", "navy": "navy blue", "offwhite": "off white", "off-white": "off white"}
# Brand names that are ordinary words; matching them as filters would misfire.
_COMMON_WORD_BRANDS = {"max", "only", "and", "w", "people", "head", "iris", "rare", "civil", "nifty",
                       "os", "noi", "nun", "iti", "pink", "life", "next", "now", "style", "trend", "go", "guess", "missy", "mish", "elle"}


# Shopping words the embedding model does not connect to catalogue vocabulary
# ("sangeet" lands on the brand "Sangria"). Appended to the search text only.
_EXPANSIONS = [
    (re.compile(r"\b(?:sangeet|mehendi|mehndi|haldi|shaadi|wedding|reception)\b", re.I),
     "festive wedding party ethnic wear lehenga sharara anarkali"),
    (re.compile(r"\b(?:office|workwear|work wear|interview|meeting|corporate|professional)\b", re.I),
     "formal work wear"),
    (re.compile(r"\bsix[- ]yards?\b", re.I), "saree"),
    (re.compile(r"\b(?:puja|pooja|diwali|eid|navratri|festival)\b", re.I), "festive ethnic traditional wear"),
    (re.compile(r"\b(?:gym|workout|yoga|running|jogging)\b", re.I), "sports activewear"),
]
# Words that carry no product meaning once filters and sort order are parsed out.
_GENERIC = frozenset("""best highest top rated rating ratings reviewed review popular loved cheapest least most
expensive lowest priced price prices cost costs affordable budget premium priciest costliest under below above over
between rupee rupees rs inr less more than within upto up max min maximum minimum at least one ones item items
product products catalogue catalog myntra have got sell available options option suggest recommend tell give get
see list compare good nice great women woman ladies lady girl girls men boy boys kids kid wear clothes clothing
outfit dress which whats how much many all only also really overall ever something anything thing things stuff pretty beautiful lovely cool""".split())


@dataclass
class Spec:
    """Constraints parsed from one (sub-)question."""
    text: str
    price_max: float | None = None
    price_min: float | None = None
    colours: list = field(default_factory=list)     # catalogue colour values
    category: str | None = None
    brands: list = field(default_factory=list)
    gender: str | None = None                       # only set for Men / Boys / Girls / Kids
    sort: str | None = None
    search_text: str = ""                           # text + domain expansions, used for retrieval
    residual: list = field(default_factory=list)    # stemmed content words not covered by a filter

    @property
    def anchored(self) -> bool:
        """True if the question names something the catalogue is organised by."""
        return bool(self.brands or self.category or self.colours)

    def describe(self) -> str:
        parts = []
        if self.brands: parts.append("brand = " + " or ".join(self.brands))
        if self.category: parts.append(f"category = {self.category}")
        if self.colours: parts.append("colour in {" + ", ".join(self.colours) + "}")
        if self.gender: parts.append(f"gender = {self.gender}")
        if self.price_min is not None: parts.append(f"price >= Rs {self.price_min:.0f}")
        if self.price_max is not None: parts.append(f"price <= Rs {self.price_max:.0f}")
        return "; ".join(parts) or "none"


def parse_prices(text: str) -> tuple[float | None, float | None]:
    lo = hi = None
    m = _RANGE_RE.search(text)
    if m:
        a, b = _amount(m.group(1), m.group(2)), _amount(m.group(3), m.group(4))
        lo, hi = min(a, b), max(a, b)
    else:
        m = _MAX_RE.search(text)
        if m:
            hi = _amount(m.group(1), m.group(2))
        m = _MIN_RE.search(text)
        if m:
            lo = _amount(m.group(1), m.group(2))
    # Small bare numbers are sizes or counts, not prices.
    lo = lo if lo is not None and lo >= 100 else None
    hi = hi if hi is not None and hi >= 100 else None
    return lo, hi


def parse_colours(text: str, catalogue_colours: list) -> list:
    """Map colour words to catalogue colour values ('blue' -> Blue, Navy Blue, ...)."""
    t = " " + re.sub(r"[^a-z ]", " ", text.lower().replace("off-white", "off white")) + " "
    for alias, real in _COLOUR_ALIASES.items():
        t = t.replace(f" {alias} ", f" {real} ")
    t = t.replace(" navy blue blue ", " navy blue ")      # "navy blue" already spelled out
    found: list = []
    for c in catalogue_colours:                       # longest names first
        cl = c.lower()
        if " " in cl and f" {cl} " in t:
            found.append(c)
            t = t.replace(f" {cl} ", "  ")
    heads = {c.lower().split()[-1] for c in catalogue_colours} - _AMBIGUOUS_COLOURS
    for word in t.split():
        if word in heads:
            found += [c for c in catalogue_colours if c.lower().split()[-1] == word and c not in found]
    return found


def parse_brands(text: str, catalogue_brands: list) -> list:
    found, masked = [], text
    for b in catalogue_brands:                        # longest names first
        if b.lower() in _COMMON_WORD_BRANDS or len(b) < 2:
            continue
        flags = re.I if len(b) >= 4 else 0             # very short names must match case exactly
        rx = re.compile(r"(?<![\w&])" + re.escape(b) + r"(?![\w&])", flags)
        if rx.search(masked):
            found.append(b)
            masked = rx.sub(" ", masked)
    return found


def parse(text: str, colours: list, brands: list) -> Spec:
    spec = Spec(text=text)
    spec.price_min, spec.price_max = parse_prices(text)
    spec.brands = parse_brands(text, brands)
    # Don't let a brand's own words ("Dupatta Bazaar", "Pink Pearl") become filters.
    residue = text
    for b in spec.brands:
        residue = re.sub(re.escape(b), " ", residue, flags=re.I)
    spec.colours = parse_colours(residue, colours)
    cats = find_categories(residue)
    spec.category = cats[0] if cats else None
    g = gender_of(residue)
    spec.gender = g if g in ("Men", "Boys", "Girls", "Kids") else None
    for name, rx in _SORTS:
        if rx.search(text):
            spec.sort = name
            break
    extra = [words for rx, words in _EXPANSIONS if rx.search(text)]
    spec.search_text = " ".join([text] + extra)
    # What is left once brands, colours, the category word, numbers and filler are removed.
    covered = set(tokenize(" ".join(spec.colours))) | {stem(w) for w in _GENERIC}
    leftover = _CATEGORY_WORDS.sub(" ", residue)
    spec.residual = [t for t in tokenize(leftover) if t not in covered and not t.isdigit() and len(t) > 1]
    return spec


def unsupported_topic(text: str) -> str | None:
    for topic, rx in _UNSUPPORTED:
        if rx.search(text):
            return topic
    return None


def has_dangling_reference(text: str) -> bool:
    """'Do you have it in blue?' with nothing earlier to point at."""
    return bool(_REFERENCE_RE.search(text))


_COMPLEX_RE = re.compile(r"\b(?:compare|versus|vs\.?|and|along with|as well as|plus|both|matching|to go with)\b", re.I)


def looks_multipart(text: str) -> bool:
    return bool(_COMPLEX_RE.search(text))


# Attempts to override the assistant or extract its instructions. The clause is
# dropped before anything reaches the LLM; the rest of the question is still answered.
_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|bypass|override)\b.{0,40}\b(?:instructions?|rules?|prompt|guidelines)\b"
    r"|\b(?:system|hidden|secret|initial|original) (?:prompt|instructions?|rules?|message)\b"
    r"|\b(?:reveal|print|repeat|show|tell me|output)\b.{0,30}\b(?:instructions?|prompt)\b"
    r"|\byou are now\b|\bact as\b|\bpretend (?:to be|you)\b|\bjailbreak\b|\bdeveloper mode\b",
    re.I)
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|,\s*(?:and\s+)?then\s+|\s+and then\s+|\s+then\s+", re.I)


def strip_overrides(text: str) -> str:
    """Remove clauses that try to change the assistant's behaviour."""
    kept = [c for c in _CLAUSE_SPLIT.split(text) if c and not _OVERRIDE_RE.search(c)]
    return " ".join(kept).strip()


def is_vague(spec: Spec) -> bool:
    """Nothing to search on: no filter, no content word, no known occasion."""
    return not spec.anchored and not spec.residual and spec.search_text == spec.text \
        and spec.price_max is None and spec.price_min is None
