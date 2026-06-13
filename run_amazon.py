#!/usr/bin/env python3
"""
Amazon All_Beauty: lexicon-based multi-label sentiment, D2F + MLNB, product search.

Labels use pretrained English lexicons (VADER + Hu & Liu opinion words) with
aspect cues mapped from SemEval-2014 ABSA / product-review literature.

Same pattern as run_scene.py: 30% hold-out, 5 runs (seed 259016+), D2F, MLNB.
Then train on all reviews and rank products by predicted aspect sentiment.

Run:
  python run_amazon.py
  python run_amazon.py --aspect quality --top 20
  python run_amazon.py --title "moisturizer" --aspect overall --top 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

from amazon_store import PRODUCTS_STORE, export_product_store
from cache_util import cache_file, config_fingerprint, data_fingerprint, load_cache, save_cache
from d2f import d2f
from sentiment_lexicon import (
    ASPECT_CUES,
    aspect_compound_bin,
    ensure_lexicons,
    hu_liu_counts,
    label_review as lexicon_label_review,
    vader_scores,
)

# --- paths ---
AMAZON_DIR = Path("data/amazon")
REVIEWS_CSV = AMAZON_DIR / "amazon_reviews_all_beauty.csv"
META_CSV = AMAZON_DIR / "amazon_meta_all_beauty.csv"
AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_CATEGORY = "All_Beauty"
AMAZON_N_REVIEWS = 2000

# --- eval config (matches run_scene.py) ---
ASPECTS = ["quality", "price", "shipping"]
LABEL_COLUMNS = [f"{a}_{p}" for a in ASPECTS for p in ("pos", "neg")]
TEST_SIZE = 0.3
N_RUNS = 5
SEED = 259016
N_SELECT = 25  # review features are fewer than Scene's 294
RULES_VERSION = "vader_huliu_v1"


# ========== download ==========

def validate_amazon() -> bool:
    if not REVIEWS_CSV.exists() or not META_CSV.exists():
        return False
    try:
        rev = pd.read_csv(REVIEWS_CSV, nrows=101)
        meta = pd.read_csv(META_CSV, nrows=101)
        return (
            {"parent_asin", "rating", "title", "text"}.issubset(rev.columns)
            and {"parent_asin", "title", "average_rating"}.issubset(meta.columns)
            and len(rev) >= 100
        )
    except Exception:
        return False


def download_amazon() -> None:
    from huggingface_hub import hf_hub_download

    AMAZON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Amazon {AMAZON_CATEGORY} reviews ({AMAZON_N_REVIEWS}) ...")
    local_jsonl = hf_hub_download(
        AMAZON_REPO, f"raw/review_categories/{AMAZON_CATEGORY}.jsonl", repo_type="dataset"
    )
    rows: list[dict] = []
    with open(local_jsonl, encoding="utf-8") as fp:
        for line in fp:
            rows.append(json.loads(line))
            if len(rows) >= AMAZON_N_REVIEWS:
                break
    reviews = pd.DataFrame(rows)
    reviews.to_csv(REVIEWS_CSV, index=False)

    print("Downloading product metadata ...")
    local_parquet = hf_hub_download(
        AMAZON_REPO,
        f"raw_meta_{AMAZON_CATEGORY}/full-00000-of-00001.parquet",
        repo_type="dataset",
    )
    meta = pd.read_parquet(local_parquet)
    asins = reviews["parent_asin"].dropna().unique()
    meta[meta["parent_asin"].isin(asins)].to_csv(META_CSV, index=False)
    print(f"Saved {REVIEWS_CSV} ({len(reviews)} reviews), {META_CSV}")


def ensure_amazon() -> None:
    if validate_amazon():
        n = len(pd.read_csv(REVIEWS_CSV))
        print(f"Amazon OK: {REVIEWS_CSV} ({n} reviews), {META_CSV}")
        return
    print("Amazon missing or invalid — downloading ...")
    download_amazon()
    if not validate_amazon():
        raise RuntimeError("Amazon download failed validation")


# ========== labels & features ==========

def _bin_count(n: int) -> int:
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    return 2


def _compound_bin(compound: float) -> int:
    if compound <= -0.5:
        return 0
    if compound <= -0.05:
        return 1
    if compound < 0.05:
        return 2
    if compound < 0.5:
        return 3
    return 4


def label_review(text: str) -> dict[str, int]:
    return lexicon_label_review(text, ASPECTS)


def label_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    combined = (
        reviews.get("title", pd.Series("", index=reviews.index)).fillna("").astype(str)
        + " "
        + reviews.get("text", pd.Series("", index=reviews.index)).fillna("").astype(str)
    )
    label_df = pd.DataFrame([label_review(t) for t in combined], index=reviews.index)
    return pd.concat([reviews, label_df], axis=1)


def _text_len_bin(length: int) -> int:
    if length < 100:
        return 0
    if length < 300:
        return 1
    if length < 600:
        return 2
    return 3


def build_features(reviews: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    combined = (
        reviews.get("title", pd.Series("", index=reviews.index)).fillna("").astype(str)
        + " "
        + reviews.get("text", pd.Series("", index=reviews.index)).fillna("").astype(str)
    )

    cols: dict[str, np.ndarray] = {}
    n = len(combined)

    vader_compound = np.zeros(n, dtype=np.int64)
    vader_pos_bin = np.zeros(n, dtype=np.int64)
    vader_neg_bin = np.zeros(n, dtype=np.int64)
    hu_pos_bin = np.zeros(n, dtype=np.int64)
    hu_neg_bin = np.zeros(n, dtype=np.int64)

    for i, text in enumerate(combined):
        scores = vader_scores(text)
        vader_compound[i] = _compound_bin(scores["compound"])
        vader_pos_bin[i] = _bin_count(int(round(scores["pos"] * 10)))
        vader_neg_bin[i] = _bin_count(int(round(scores["neg"] * 10)))
        hu_pos, hu_neg = hu_liu_counts(text)
        hu_pos_bin[i] = _bin_count(hu_pos)
        hu_neg_bin[i] = _bin_count(hu_neg)

    cols["vader_compound_bin"] = vader_compound
    cols["vader_pos_bin"] = vader_pos_bin
    cols["vader_neg_bin"] = vader_neg_bin
    cols["hu_liu_pos_bin"] = hu_pos_bin
    cols["hu_liu_neg_bin"] = hu_neg_bin

    for aspect in ASPECTS:
        cue_hits = np.array(
            [sum(cue in t.lower() for cue in ASPECT_CUES[aspect]) for t in combined],
            dtype=np.int64,
        )
        cols[f"cue_{aspect}_bin"] = np.array([_bin_count(int(h)) for h in cue_hits], dtype=np.int64)
        cols[f"vader_{aspect}_bin"] = np.array(
            [aspect_compound_bin(t, aspect) for t in combined], dtype=np.int64
        )

    if "rating" in reviews.columns:
        ratings = pd.to_numeric(reviews["rating"], errors="coerce").fillna(3).astype(int)
        cols["rating"] = (ratings.clip(1, 5) - 1).astype(np.int64).values
    cols["text_len_bin"] = np.array([_text_len_bin(len(t)) for t in combined], dtype=np.int64)

    names = list(cols.keys())
    return np.column_stack([cols[n] for n in names]).astype(np.int64), names


# ========== MLNB (same as run_scene.py) ==========

def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))


def multilabel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def fit_mlnb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_categories: np.ndarray,
) -> list[CategoricalNB]:
    return [
        CategoricalNB(min_categories=min_categories).fit(x_train, y_train[:, j])
        for j in range(y_train.shape[1])
    ]


def predict_mlnb(
    models: list[CategoricalNB],
    x: np.ndarray,
    min_categories: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64).copy()
    for j in range(x.shape[1]):
        x[:, j] = np.clip(x[:, j], 0, int(min_categories[j]) - 1)
    return np.column_stack([m.predict(x) for m in models]).astype(int)


def evaluate_mlnb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    min_categories: np.ndarray,
) -> tuple[float, float]:
    models = fit_mlnb(x_train, y_train, min_categories)
    y_pred = predict_mlnb(models, x_test, min_categories)
    return hamming_loss(y_test, y_pred), multilabel_accuracy(y_test, y_pred)


# ========== train / eval / rank ==========

def run_eval(
    x: np.ndarray,
    y: np.ndarray,
    n_select: int,
    min_categories: np.ndarray,
) -> tuple[float, float, float, float]:
    hamming_runs: list[float] = []
    acc_runs: list[float] = []
    for run in range(N_RUNS):
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=SEED + run
        )
        k = min(n_select, x_tr.shape[1])
        selected = d2f(x_tr, y_tr, k, progress=(run == 0))
        h, acc = evaluate_mlnb(
            x_tr[:, selected], y_tr, x_te[:, selected], y_te, min_categories[selected]
        )
        hamming_runs.append(h)
        acc_runs.append(acc)
        print(f"  Run {run + 1}/{N_RUNS} (seed {SEED + run}): Hamming {h:.4f} | Accuracy {acc:.4f}")
    return (
        float(np.mean(hamming_runs)),
        float(np.std(hamming_runs, ddof=0)),
        float(np.mean(acc_runs)),
        float(np.std(acc_runs, ddof=0)),
    )


def _amazon_config(n_select: int) -> dict:
    return {
        "test_size": TEST_SIZE,
        "n_runs": N_RUNS,
        "seed": SEED,
        "n_select": n_select,
        "rules_version": RULES_VERSION,
        "label_columns": LABEL_COLUMNS,
    }


def _load_or_train_amazon(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int,
    min_categories: np.ndarray,
    *,
    refresh: bool,
) -> tuple[dict, dict, float]:
    data_fp = data_fingerprint(REVIEWS_CSV)
    cfg_fp = config_fingerprint(_amazon_config(n_select))
    eval_path = cache_file("amazon_eval", data_fp, cfg_fp)
    model_path = cache_file("amazon_model", data_fp, cfg_fp)

    eval_cache = None if refresh else load_cache(eval_path)
    model_cache = None if refresh else load_cache(model_path)

    t0 = time.perf_counter()
    if eval_cache:
        print(f"Using cached eval: {eval_path.name}")
    else:
        h_mean, h_std, acc_mean, acc_std = run_eval(x, y, n_select, min_categories)
        eval_cache = {
            "h_mean": h_mean,
            "h_std": h_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }
        save_cache(eval_path, eval_cache)
        print(f"Cached eval -> {eval_path}")

    if model_cache:
        print(f"Using cached model: {model_path.name}")
    else:
        print("\n=== Final model (all reviews) ===")
        selected = d2f(x, y, n_select, progress=True)
        models = fit_mlnb(x[:, selected], y, min_categories[selected])
        model_cache = {
            "selected": selected,
            "models": models,
            "min_categories_selected": min_categories[selected],
            "feature_names": feature_names,
        }
        save_cache(model_path, model_cache)
        print(f"Cached model -> {model_path}")

    elapsed = time.perf_counter() - t0
    return eval_cache, model_cache, elapsed


def aspect_score_from_preds(group: pd.DataFrame, aspect: str) -> float:
    pos = group[f"{aspect}_pos"].mean()
    neg = group[f"{aspect}_neg"].mean()
    return float(pos - neg)


def build_product_scores(
    reviews: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[CategoricalNB],
    selected: np.ndarray,
    x: np.ndarray,
    min_categories: np.ndarray,
) -> pd.DataFrame:
    """Predict per-review labels and aggregate to per-product sentiment scores."""
    sel_cats = min_categories[selected]
    preds = predict_mlnb(models, x[:, selected], sel_cats)
    pred_df = reviews[["parent_asin", "rating"]].copy()
    for j, col in enumerate(LABEL_COLUMNS):
        pred_df[col] = preds[:, j]

    rows = []
    for asin, group in pred_df.groupby("parent_asin"):
        row: dict = {
            "parent_asin": asin,
            "review_count": len(group),
            "mean_rating": float(group["rating"].mean()) if "rating" in group.columns else np.nan,
        }
        for a in ASPECTS:
            row[f"sentiment_{a}"] = aspect_score_from_preds(group, a)
        row["overall_sentiment"] = float(np.mean([row[f"sentiment_{a}"] for a in ASPECTS]))
        rows.append(row)

    scores = pd.DataFrame(rows)
    meta_cols = ["parent_asin", "title", "average_rating", "price"]
    meta_cols = [c for c in meta_cols if c in meta.columns]
    meta_sub = meta[meta_cols].drop_duplicates("parent_asin")
    scores = scores.merge(meta_sub, on="parent_asin", how="left")
    scores = scores.rename(columns={"title": "product_title", "average_rating": "meta_rating"})
    if "price" in scores.columns:
        scores["price"] = pd.to_numeric(scores["price"], errors="coerce")
    return scores


def rank_products(
    reviews: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[CategoricalNB],
    selected: np.ndarray,
    x: np.ndarray,
    min_categories: np.ndarray,
    aspect: str,
    top: int,
    title_filter: str | None,
) -> pd.DataFrame:
    scores = build_product_scores(reviews, meta, models, selected, x, min_categories)

    sort_col = "overall_sentiment" if aspect == "overall" else f"sentiment_{aspect}"
    ranked = scores.sort_values(sort_col, ascending=False)
    if title_filter:
        mask = ranked["product_title"].fillna("").str.contains(title_filter, case=False, na=False)
        ranked = ranked[mask]
    return ranked.head(top).reset_index(drop=True)


def print_ranking(ranked: pd.DataFrame, aspect: str) -> None:
    sort_col = "overall_sentiment" if aspect == "overall" else f"sentiment_{aspect}"
    cols = ["product_title", sort_col, "sentiment_quality", "sentiment_price", "sentiment_shipping",
            "mean_rating", "review_count", "parent_asin"]
    cols = list(dict.fromkeys(c for c in cols if c in ranked.columns))
    print(ranked[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon D2F + MLNB sentiment ranking")
    parser.add_argument(
        "--aspect",
        default="overall",
        choices=["overall", "quality", "price", "shipping"],
        help="Rank products by this sentiment aspect",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of products to show")
    parser.add_argument("--title", type=str, default=None, help="Filter product titles (substring)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and retrain")
    args = parser.parse_args()

    print("=== Data ===")
    ensure_amazon()
    ensure_lexicons()
    if args.download_only:
        print("Download complete.")
        return

    reviews = pd.read_csv(REVIEWS_CSV)
    meta = pd.read_csv(META_CSV)
    labeled = label_reviews(reviews)
    x, feature_names = build_features(labeled)
    y = labeled[LABEL_COLUMNS].astype(int).values
    n_select = min(N_SELECT, x.shape[1])
    min_categories = np.maximum(x.max(axis=0) + 1, 2)

    coverage = (labeled[LABEL_COLUMNS].sum(axis=1) > 0).mean()
    print(f"\n=== Amazon sentiment (D2F + MLNB) ===")
    print(f"Reviews: {len(reviews)} | Features: {x.shape[1]} | Labels: {y.shape[1]}")
    print(f"Rule-label coverage (>=1 aspect): {coverage:.1%}")
    print("Lexicons: VADER (2014) + Hu & Liu opinion_lexicon (NLTK) | Aspect cues: SemEval-2014 ABSA mapping")
    print(f"Split: 70/30 hold-out | Runs: {N_RUNS} | Seeds: {SEED}..{SEED + N_RUNS - 1}")
    print(f"D2F features to select: {n_select} | Classifier: MLNB")
    if args.refresh:
        print("Cache: disabled (--refresh)\n")
    else:
        print("Cache: data/cache (use --refresh to retrain)\n")

    eval_cache, model_cache, elapsed = _load_or_train_amazon(
        x, y, feature_names, n_select, min_categories, refresh=args.refresh
    )
    print(
        f"\nEval mean: Hamming {eval_cache['h_mean']:.4f} (±{eval_cache['h_std']:.4f}) | "
        f"Accuracy {eval_cache['acc_mean']:.4f} (±{eval_cache['acc_std']:.4f})"
    )

    selected = model_cache["selected"]
    models = model_cache["models"]
    print(f"\nModel ready ({len(selected)} D2F features, loaded in {elapsed:.1f}s)")
    print(
        f"  Features: {[feature_names[i] for i in selected[:8]]}"
        f"{'...' if len(selected) > 8 else ''}"
    )

    ranked = rank_products(
        labeled, meta, models, selected, x, min_categories, args.aspect, args.top, args.title
    )
    title_note = f" matching '{args.title}'" if args.title else ""
    print(f"\n=== Top {args.top} by {args.aspect} sentiment{title_note} ===")
    print_ranking(ranked, args.aspect)

    all_products = build_product_scores(labeled, meta, models, selected, x, min_categories)
    store_path = export_product_store(
        all_products,
        version=RULES_VERSION,
        eval_stats={
            "hamming_mean": eval_cache["h_mean"],
            "accuracy_mean": eval_cache["acc_mean"],
        },
        path=PRODUCTS_STORE,
    )
    print(f"\nProduct store: {store_path} ({len(all_products)} products)")
    print("Start API: uvicorn amazon_api:app --reload")
    print("\nDone.")


if __name__ == "__main__":
    main()
