"""
Pretrained English sentiment lexicons (no ML training, no GPU).

- VADER (Hutto & Gilbert, 2014): rule-based polarity with negation/intensifiers.
- Hu & Liu opinion lexicon (2004): ~6800 positive/negative words via NLTK.

Aspect cues map SemEval-2014 ABSA categories + product-review literature to
quality / price / shipping (indicators only; polarity from VADER + Hu-Liu).
"""

from __future__ import annotations

import re
from functools import lru_cache

# Aspect indicators (not sentiment phrases). Sources:
# - SemEval-2014 Task 4 (ABSA): PRICE; product-quality nouns from laptop/restaurant tasks
# - Hu & Liu (KDD-2004) customer-review feature vocabulary (delivery, packaging)
ASPECT_CUES: dict[str, list[str]] = {
    "quality": [
        "quality", "durable", "durability", "material", "materials", "sturdy", "flimsy",
        "works", "worked", "working", "performance", "effective", "effectiveness",
        "texture", "scent", "smell", "formula", "ingredient", "results", "lasted",
        "broke", "broken", "defective", "disappointing",
    ],
    "price": [
        "price", "priced", "cost", "costs", "value", "worth", "money", "expensive",
        "cheap", "affordable", "overpriced", "bargain", "deal", "budget",
    ],
    "shipping": [
        "delivery", "delivered", "shipping", "shipped", "arrived", "arrive", "package",
        "packaging", "packaged", "box", "courier", "postage", "dispatch",
    ],
}

VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def ensure_lexicons() -> None:
    import nltk

    try:
        nltk.data.find("corpora/opinion_lexicon")
    except LookupError:
        nltk.download("opinion_lexicon", quiet=True)


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _hu_liu_words() -> tuple[frozenset[str], frozenset[str]]:
    ensure_lexicons()
    from nltk.corpus import opinion_lexicon

    return frozenset(opinion_lexicon.positive()), frozenset(opinion_lexicon.negative())


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def hu_liu_counts(text: str) -> tuple[int, int]:
    pos_w, neg_w = _hu_liu_words()
    tokens = _tokenize(text)
    pos = sum(1 for t in tokens if t in pos_w)
    neg = sum(1 for t in tokens if t in neg_w)
    return pos, neg


def vader_scores(text: str) -> dict[str, float]:
    return _vader().polarity_scores(text)


def _aspect_cue_hits(text: str, aspect: str) -> int:
    t = text.lower()
    return sum(1 for cue in ASPECT_CUES[aspect] if cue in t)


def _aspect_sentences(text: str, aspect: str) -> list[str]:
    cues = ASPECT_CUES[aspect]
    return [s for s in split_sentences(text) if any(c in s.lower() for c in cues)]


def label_aspect(text: str, aspect: str) -> tuple[int, int]:
    """Return (pos, neg) for one aspect using VADER on cue-bearing sentences."""
    sentences = _aspect_sentences(text, aspect)
    if not sentences and _aspect_cue_hits(text, aspect) == 0:
        return 0, 0
    if not sentences:
        sentences = [text]

    pos_votes = 0
    neg_votes = 0
    for sent in sentences:
        compound = vader_scores(sent)["compound"]
        if compound >= VADER_POS_THRESHOLD:
            pos_votes += 1
        elif compound <= VADER_NEG_THRESHOLD:
            neg_votes += 1
        else:
            hu_pos, hu_neg = hu_liu_counts(sent)
            if hu_pos > hu_neg:
                pos_votes += 1
            elif hu_neg > hu_pos:
                neg_votes += 1

    pos = int(pos_votes > 0 and pos_votes >= neg_votes)
    neg = int(neg_votes > 0 and neg_votes > pos_votes)
    return pos, neg


def aspect_compound_bin(text: str, aspect: str) -> int:
    sentences = _aspect_sentences(text, aspect)
    if not sentences:
        return 2  # neutral bin when aspect not discussed
    avg = sum(vader_scores(s)["compound"] for s in sentences) / len(sentences)
    if avg <= -0.5:
        return 0
    if avg <= -0.05:
        return 1
    if avg < 0.05:
        return 2
    if avg < 0.5:
        return 3
    return 4


def label_review(text: str, aspects: list[str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for aspect in aspects:
        pos, neg = label_aspect(text, aspect)
        labels[f"{aspect}_pos"] = pos
        labels[f"{aspect}_neg"] = neg
    return labels
