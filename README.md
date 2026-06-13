# D2F Feature Selection (Lee & Kim, 2015)

Minimal implementation of **D2F** validated on the **Scene** benchmark.

## Files

| File | Purpose |
|------|---------|
| [`d2f.py`](d2f.py) | D2F algorithm (port of author MATLAB code) |
| [`run_scene.py`](run_scene.py) | Download Scene → D2F + ML-kNN evaluation |
| [`run_amazon.py`](run_amazon.py) | Amazon reviews → D2F + MLNB → rank products by sentiment |
| [`amazon_api.py`](amazon_api.py) | REST API + web UI for product search |
| [`web/index.html`](web/index.html) | Demo search page (sentiment toggle) |
| [`amazon_store.py`](amazon_store.py) | Product store export/load and search helpers |
| [`main.tex`](main.tex) | Project proposal |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Downloads Scene automatically if missing. Compares our run to **Lee & Kim (2015)** Scene results (Tables 2 & 4).

```bash
python run_scene.py
python run_scene.py --refresh   # ignore cache, retrain
python run_scene.py --run 0     # single split only (for parallel)
python run_scene.py --merge-only
./run_scene_parallel.sh         # all 5 runs in parallel (~wall-clock of 1 run)
./run_scene_parallel.sh --refresh
```

**Our protocol:** 30% hold-out, 5 runs (seeds 259016–259020), 3 bins, **ML-kNN (k=10)**, D2F selects 50 features.

**Paper protocol (§5.1):** same split ratio, **100-run** average, **MLNB**, best subset in 1–50.

**Paper Scene values (D2F + MLNB, Table 4):** Hamming **0.1288**, accuracy **0.5500** (reference only; we report ML-kNN scores).

Exact match to Table 4 is not guaranteed (paper averages 100 runs and picks best n in 1–50). D2F selection per run is slow on CPU (~10–15 min × 5 runs sequential). Use `./run_scene_parallel.sh` to run all 5 splits in parallel (one process each). Progress is cached per run and during D2F select — safe to interrupt; use `--refresh` to start over.

### Amazon sentiment ranking

```bash
python run_amazon.py
python run_amazon.py --aspect quality --top 20
python run_amazon.py --refresh   # ignore cache, retrain
```

Uses **VADER** + **Hu & Liu opinion lexicon** (NLTK) for polarity, with aspect cues mapped from SemEval-2014 ABSA to quality / price / shipping. Same **30% hold-out, 5 runs, MLNB, D2F** pattern, then trains on all reviews and ranks products by predicted sentiment. Writes **`data/amazon/products_store.json`** for the REST API. Results cached under `data/cache/` (invalidates if data or config changes).

### REST API (search + sort)

After training, start the API:

```bash
pip install -r requirements.txt
python run_amazon.py              # builds products_store.json
uvicorn amazon_api:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Store loaded? version, product count |
| `GET /api/sort-options` | Available sort fields |
| `GET /api/products` | Search + sort (see query params below) |
| `GET /api/products/{asin}` | Single product by ASIN |
| `POST /api/reload` | Reload store after retraining |

**Query params for `/api/products`:**

| Param | Example | Notes |
|-------|---------|-------|
| `query` | `moisturizer` | Substring match on title |
| `sort_by` | `meta_rating` | Also: `price`, `review_count`, `title_relevance`, `sentiment_quality`, … |
| `use_sentiment` | `true` / `false` | Toggle sentiment-based sort vs star-rating sort |
| `order` | `desc` | `asc` or `desc` |
| `limit` | `20` | Max 100 |

Examples:

```bash
# Sort by star rating (traditional)
curl 'http://127.0.0.1:8000/api/products?query=serum&sort_by=meta_rating&use_sentiment=false'

# Sort by quality sentiment from reviews
curl 'http://127.0.0.1:8000/api/products?query=serum&sort_by=sentiment_quality&use_sentiment=true'
```

Open **http://127.0.0.1:8000/** for the demo web UI (search + sentiment sort toggle), or **http://127.0.0.1:8000/docs** for Swagger.

## Data

| Dataset | Path |
|---------|------|
| Scene | `data/raw/scene/scene.arff` |
| Amazon reviews | `data/amazon/amazon_reviews_all_beauty.csv` |
| Amazon metadata | `data/amazon/amazon_meta_all_beauty.csv` |
