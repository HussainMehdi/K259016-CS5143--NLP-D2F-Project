"""
REST API for Amazon product search with optional sentiment-based sorting.

Prerequisites:
  python run_amazon.py          # trains model + writes data/amazon/products_store.json
  uvicorn amazon_api:app --reload --port 8000

Examples:
  GET /health
  GET /api/products?query=moisturizer&sort_by=meta_rating&use_sentiment=false
  GET /api/products?query=moisturizer&sort_by=sentiment_quality&use_sentiment=true
  GET /api/sort-options
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from amazon_store import (
    PRODUCTS_STORE,
    SENTIMENT_SORT_FIELDS,
    SORT_FIELDS,
    load_product_store,
    search_products,
    store_summary,
)

_store: dict[str, Any] | None = None
WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _store
    try:
        _store = load_product_store()
    except FileNotFoundError:
        _store = None
    yield


app = FastAPI(
    title="D2F-Rank Amazon Search",
    description="Search beauty products; sort by ratings, price, or review sentiment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_store() -> dict[str, Any]:
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail=f"Product store missing. Run: python run_amazon.py (expected {PRODUCTS_STORE})",
        )
    return _store


@app.get("/")
def web_ui() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(index)


@app.get("/health")
def health() -> dict[str, Any]:
    if _store is None:
        return {"status": "degraded", "store_loaded": False, "store_path": str(PRODUCTS_STORE)}
    return {"status": "ok", "store_loaded": True, **store_summary(_store)}


@app.post("/api/reload")
def reload_store() -> dict[str, Any]:
    """Reload products_store.json from disk (after re-running run_amazon.py)."""
    global _store
    try:
        _store = load_product_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "reloaded", **store_summary(_store)}


@app.get("/api/sort-options")
def sort_options() -> dict[str, Any]:
    return {
        "traditional": sorted(SORT_FIELDS - SENTIMENT_SORT_FIELDS),
        "sentiment": sorted(SENTIMENT_SORT_FIELDS),
        "use_sentiment": (
            "When true, sort_by must be a sentiment_* field (defaults to overall_sentiment). "
            "When false, sentiment fields are returned but sort uses ratings/price/relevance."
        ),
    }


@app.get("/api/products")
def list_products(
    query: str | None = Query(None, description="Substring filter on product title"),
    sort_by: str = Query(
        "meta_rating",
        description="Sort field: meta_rating, mean_rating, price, review_count, "
        "title_relevance, overall_sentiment, sentiment_quality, sentiment_price, sentiment_shipping",
    ),
    use_sentiment: bool = Query(
        False,
        description="If true, sort by sentiment (sort_by must be sentiment_* or overall_sentiment)",
    ),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(20, ge=1, le=100),
    min_reviews: int = Query(1, ge=1),
) -> dict[str, Any]:
    store = _require_store()
    results = search_products(
        store,
        query=query,
        sort_by=sort_by,
        use_sentiment=use_sentiment,
        order=order,
        limit=limit,
        min_reviews=min_reviews,
    )
    effective_sort = sort_by
    if use_sentiment and sort_by not in SENTIMENT_SORT_FIELDS:
        effective_sort = "overall_sentiment"
    elif not use_sentiment and sort_by in SENTIMENT_SORT_FIELDS:
        effective_sort = "meta_rating"
    elif sort_by == "title_relevance" and not (query or "").strip():
        effective_sort = "meta_rating"

    return {
        "query": query,
        "sort_by": effective_sort,
        "use_sentiment": use_sentiment,
        "order": order,
        "count": len(results),
        "products": results,
    }


@app.get("/api/products/{parent_asin}")
def get_product(parent_asin: str) -> dict[str, Any]:
    store = _require_store()
    for product in store.get("products", []):
        if product.get("parent_asin") == parent_asin:
            return product
    raise HTTPException(status_code=404, detail=f"Product {parent_asin} not found")
