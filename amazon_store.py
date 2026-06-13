"""
Product sentiment store: export/load JSON catalog and search/sort for the REST API.

Build with `run_amazon.py` (writes data/amazon/products_store.json).
The API loads that file — no retraining at request time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRODUCTS_STORE = Path("data/amazon/products_store.json")

SORT_FIELDS = frozenset(
    {
        "title_relevance",
        "meta_rating",
        "mean_rating",
        "price",
        "review_count",
        "overall_sentiment",
        "sentiment_quality",
        "sentiment_price",
        "sentiment_shipping",
    }
)

SENTIMENT_SORT_FIELDS = frozenset(
    {
        "overall_sentiment",
        "sentiment_quality",
        "sentiment_price",
        "sentiment_shipping",
    }
)


def export_product_store(
    products: pd.DataFrame,
    *,
    version: str,
    eval_stats: dict[str, float] | None = None,
    path: Path = PRODUCTS_STORE,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = products.replace({np.nan: None}).to_dict(orient="records")
    payload = {
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "product_count": len(records),
        "eval": eval_stats or {},
        "products": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_product_store(path: Path = PRODUCTS_STORE) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Product store not found at {path}. Run: python run_amazon.py"
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def title_relevance(title: str | None, query: str) -> float:
    if not query or not query.strip():
        return 0.0
    title_l = (title or "").lower()
    query_l = query.strip().lower()
    if not title_l:
        return 0.0
    if query_l in title_l:
        return 1.0 + len(query_l) / max(len(title_l), 1)
    tokens = [t for t in query_l.split() if len(t) > 1]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in title_l)
    return hits / len(tokens)


def search_products(
    store: dict[str, Any],
    *,
    query: str | None = None,
    sort_by: str = "meta_rating",
    use_sentiment: bool = False,
    order: str = "desc",
    limit: int = 20,
    min_reviews: int = 1,
) -> list[dict[str, Any]]:
    products = [dict(p) for p in store.get("products", [])]
    q = (query or "").strip()

    if q:
        products = [
            p
            for p in products
            if q.lower() in (p.get("product_title") or "").lower()
        ]

    for product in products:
        product["title_relevance"] = title_relevance(product.get("product_title"), q)

    products = [p for p in products if int(p.get("review_count") or 0) >= min_reviews]

    if use_sentiment:
        if sort_by not in SENTIMENT_SORT_FIELDS:
            sort_by = "overall_sentiment"
    else:
        if sort_by in SENTIMENT_SORT_FIELDS:
            sort_by = "meta_rating"
        if sort_by == "title_relevance" and not q:
            sort_by = "meta_rating"

    if sort_by not in SORT_FIELDS:
        sort_by = "meta_rating"

    reverse = order.lower() != "asc"

    def sort_key(item: dict[str, Any]) -> tuple:
        value = item.get(sort_by)
        if value is None:
            return (1, 0.0)
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, 0.0)

    products.sort(key=sort_key, reverse=reverse)
    return products[: max(1, min(limit, 100))]


def store_summary(store: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": store.get("version"),
        "built_at": store.get("built_at"),
        "product_count": store.get("product_count"),
        "eval": store.get("eval", {}),
    }
