#!/usr/bin/env python3
"""
D2F on Scene benchmark (Lee & Kim 2015).

Eval: 30% hold-out, 3 bins, ML-kNN (k=10), D2F selects N_SELECT features, 5 runs averaged.
Seeds: SEED, SEED+1, ... (paper §5.1 uses 100 runs; we use 5 for speed).

Run:
  python run_scene.py                    # sequential (all runs)
  python run_scene.py --run 0            # single run (for parallel)
  python run_scene.py --merge-only       # combine per-run caches + print results
  ./run_scene_parallel.sh                # all runs in parallel
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer

from cache_util import cache_file, config_fingerprint, data_fingerprint, load_cache, save_cache
from d2f import d2f

# --- paths ---
SCENE_ARFF = Path("data/raw/scene/scene.arff")
SCENE_URLS = [
    "https://datahub.io/core/openml-datasets/_r/-/data/scene/scene.arff",
    "https://raw.githubusercontent.com/mrapp-ke/Boomer-Datasets/master/scene/scene.arff",
]

# --- evaluation config (Lee & Kim 2015 §5.1) ---
LABELS = ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]
N_BINS = 3
N_SELECT = 50
TEST_SIZE = 0.3
N_RUNS = 5
SEED = 259016
MLKNN_K = 10

LEE_KIM_2015_SCENE = {
    "source": "Lee & Kim (2015) Table 4",
    "paper_classifier": "MLNB",
    "our_classifier": f"ML-kNN (k={MLKNN_K})",
    "paper_split": "30% hold-out (100-run average)",
    "d2f": {"hamming": 0.1288, "accuracy": 0.5500},
}


# ========== data ==========

def load_scene(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw, _ = arff.loadarff(str(path))
    df = pd.DataFrame(raw)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    label_cols = [c for c in LABELS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in label_cols]
    return df[feat_cols].astype(float).values, df[label_cols].astype(int).values


def validate_scene(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        x, y = load_scene(path)
        return x.shape[0] >= 2000 and x.shape[1] >= 200 and y.shape[1] == 6
    except Exception:
        return False


def download_scene(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in SCENE_URLS:
        print(f"Downloading Scene from {url} ...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            if b"@relation" in resp.content.lower():
                path.write_bytes(resp.content)
                print(f"Saved {path}")
                return
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
    raise RuntimeError("Could not download Scene dataset")


def ensure_scene() -> None:
    if validate_scene(SCENE_ARFF):
        print(f"Scene OK: {SCENE_ARFF}")
        return
    print("Scene missing or invalid — downloading ...")
    download_scene(SCENE_ARFF)
    if not validate_scene(SCENE_ARFF):
        raise RuntimeError("Scene download failed validation")
    print("Scene validated.")


# ========== evaluation ==========

def discretize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="uniform")
    return disc.fit_transform(x_train).astype(np.int64), disc.transform(x_test).astype(np.int64)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))


def multilabel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples with all labels predicted correctly (higher = better)."""
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def mlknn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    k: int = MLKNN_K,
) -> np.ndarray:
    """ML-kNN: k nearest train samples; label = majority vote per label (MULAN / CRMIL protocol)."""
    k = min(k, len(x_train))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_train)
    _, indices = nn.kneighbors(x_test)
    preds = np.zeros((len(x_test), y_train.shape[1]), dtype=int)
    for i, neigh_idx in enumerate(indices):
        preds[i] = (y_train[neigh_idx].mean(axis=0) >= 0.5).astype(int)
    return preds


def evaluate_mlknn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    y_pred = mlknn_predict(x_train, y_train, x_test)
    return hamming_loss(y_test, y_pred), multilabel_accuracy(y_test, y_pred)


def run_single_split(
    x: np.ndarray,
    y: np.ndarray,
    run_idx: int,
    *,
    show_d2f_progress: bool,
    d2f_checkpoint: Path | None = None,
) -> tuple[float, float, np.ndarray]:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED + run_idx
    )
    x_tr_d, x_te_d = discretize(x_tr, x_te)
    selected = d2f(
        x_tr_d, y_tr, N_SELECT, progress=show_d2f_progress, checkpoint=d2f_checkpoint
    )
    h, acc = evaluate_mlknn(x_tr_d[:, selected], y_tr, x_te_d[:, selected], y_te)
    return h, acc, selected


def _scene_config() -> dict:
    return {
        "n_bins": N_BINS,
        "n_select": N_SELECT,
        "test_size": TEST_SIZE,
        "n_runs": N_RUNS,
        "seed": SEED,
        "labels": LABELS,
        "classifier": "mlknn",
        "mlknn_k": MLKNN_K,
    }


def _benchmark_payload(
    hamming_runs: list[float],
    accuracy_runs: list[float],
    last_selected: np.ndarray | None,
) -> dict:
    return {
        "hamming_runs": hamming_runs,
        "accuracy_runs": accuracy_runs,
        "last_selected": last_selected.tolist() if last_selected is not None else None,
        "completed_runs": len(hamming_runs),
    }


def _run_cache_path(data_fp: str, cfg_fp: str, run_idx: int) -> Path:
    return cache_file(f"scene_run{run_idx}", data_fp, cfg_fp)


def _d2f_cache_path(data_fp: str, cfg_fp: str, run_idx: int) -> Path:
    return cache_file(f"scene_d2f_run{run_idx}", data_fp, cfg_fp)


def _cache_paths() -> tuple[str, str, Path]:
    data_fp = data_fingerprint(SCENE_ARFF)
    cfg_fp = config_fingerprint(_scene_config())
    benchmark_path = cache_file("scene_benchmark", data_fp, cfg_fp)
    return data_fp, cfg_fp, benchmark_path


def _clear_run_cache(data_fp: str, cfg_fp: str, run_idx: int) -> None:
    _run_cache_path(data_fp, cfg_fp, run_idx).unlink(missing_ok=True)
    _d2f_cache_path(data_fp, cfg_fp, run_idx).unlink(missing_ok=True)


def run_one(
    x: np.ndarray,
    y: np.ndarray,
    run_idx: int,
    data_fp: str,
    cfg_fp: str,
    *,
    refresh: bool = False,
    show_d2f_progress: bool = True,
) -> tuple[float, float, np.ndarray]:
    """Run a single hold-out split; cache to scene_run{N}.pkl (parallel-safe)."""
    run_path = _run_cache_path(data_fp, cfg_fp, run_idx)

    if refresh:
        _clear_run_cache(data_fp, cfg_fp, run_idx)
    elif run_path.exists():
        cached = load_cache(run_path)
        if cached and cached.get("run_idx") == run_idx:
            print(
                f"Run {run_idx + 1}/{N_RUNS} (seed {SEED + run_idx}): "
                f"cached Hamming {cached['hamming']:.4f} | Accuracy {cached['accuracy']:.4f}"
            )
            return (
                float(cached["hamming"]),
                float(cached["accuracy"]),
                np.asarray(cached["selected"], dtype=np.int64),
            )

    print(f"--- Run {run_idx + 1}/{N_RUNS} (seed {SEED + run_idx}) ---")
    d2f_ckpt = _d2f_cache_path(data_fp, cfg_fp, run_idx)
    h, acc, selected = run_single_split(
        x,
        y,
        run_idx,
        show_d2f_progress=show_d2f_progress,
        d2f_checkpoint=None if refresh else d2f_ckpt,
    )
    save_cache(
        run_path,
        {
            "run_idx": run_idx,
            "hamming": h,
            "accuracy": acc,
            "selected": selected.tolist(),
        },
    )
    print(f"  Hamming {h:.4f} | Accuracy {acc:.4f}")
    print(f"  Cached -> {run_path.name}")
    return h, acc, selected


def merge_run_caches(
    data_fp: str,
    cfg_fp: str,
    benchmark_path: Path,
) -> dict | None:
    """Merge per-run caches into scene_benchmark.pkl. Returns payload or None if incomplete."""
    hamming_runs: list[float] = []
    accuracy_runs: list[float] = []
    last_selected: np.ndarray | None = None

    for run_idx in range(N_RUNS):
        run_path = _run_cache_path(data_fp, cfg_fp, run_idx)
        cached = load_cache(run_path)
        if not cached or cached.get("run_idx") != run_idx:
            return None
        hamming_runs.append(float(cached["hamming"]))
        accuracy_runs.append(float(cached["accuracy"]))
        last_selected = np.asarray(cached["selected"], dtype=np.int64)

    payload = _benchmark_payload(hamming_runs, accuracy_runs, last_selected)
    save_cache(benchmark_path, payload)
    return payload


def print_results(
    hamming_runs: list[float],
    accuracy_runs: list[float],
    last_selected: np.ndarray | None,
    *,
    elapsed: float,
) -> None:
    ref = LEE_KIM_2015_SCENE
    h_mean, h_std = float(np.mean(hamming_runs)), float(np.std(hamming_runs, ddof=0))
    acc_mean, acc_std = float(np.mean(accuracy_runs)), float(np.std(accuracy_runs, ddof=0))

    print(f"\nTotal time: {elapsed:.1f}s")
    if last_selected is not None:
        print(f"  Selected features (run {N_RUNS}): {last_selected.tolist()}")

    ref_d2f = ref["d2f"]
    print(f"\n{'':12} {'Hamming':>8} {'Accuracy':>8}  {'Paper Ham':>9} {'Paper Acc':>9}")
    print("-" * 54)
    print(
        f"{'D2F mean':<12} {h_mean:8.4f} {acc_mean:8.4f}  "
        f"{ref_d2f['hamming']:9.4f} {ref_d2f['accuracy']:9.4f}"
    )
    print(f"{'D2F std':<12} {h_std:8.4f} {acc_std:8.4f}")

    print(f"\nPer-run Hamming: {[f'{v:.4f}' for v in hamming_runs]}")
    print(f"Per-run Accuracy: {[f'{v:.4f}' for v in accuracy_runs]}")
    print(
        f"\nComparison (Scene, D2F n={N_SELECT}, {N_RUNS}-run mean; classifier: "
        f"{ref['our_classifier']}):"
    )
    print(f"  Lee & Kim Table 4 (MLNB): Hamming {ref_d2f['hamming']:.4f}, Accuracy {ref_d2f['accuracy']:.4f}")
    print(f"  Ours (ML-kNN):            Hamming {h_mean:.4f}, Accuracy {acc_mean:.4f}")


def print_benchmark_header() -> None:
    ref = LEE_KIM_2015_SCENE
    x, y = load_scene(SCENE_ARFF)
    print("\n=== D2F on Scene ===")
    print(f"Samples: {len(x)} | Features: {x.shape[1]} | Labels: {y.shape[1]}")
    print(f"Split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} hold-out | Runs: {N_RUNS}")
    print(f"Seeds: {SEED} .. {SEED + N_RUNS - 1}")
    print(f"Discretization: {N_BINS} equal-width bins | Classifier: {ref['our_classifier']}")
    print(f"D2F features to select: {N_SELECT}")
    print(f"Paper ref ({ref['source']}): {ref['paper_classifier']}, {ref['paper_split']}")
    print("Note: each run takes ~10–15 min on CPU (D2F select is slow); progress is cached per run.")


def run_benchmark(*, refresh: bool = False, run_idx: int | None = None) -> None:
    x, y = load_scene(SCENE_ARFF)
    data_fp, cfg_fp, benchmark_path = _cache_paths()

    print_benchmark_header()
    if refresh:
        print("Cache: disabled (--refresh)")
    else:
        print("Cache: data/cache — resumes after interrupt (use --refresh to restart)")

    if run_idx is not None:
        t0 = time.perf_counter()
        run_one(x, y, run_idx, data_fp, cfg_fp, refresh=refresh, show_d2f_progress=True)
        elapsed = time.perf_counter() - t0
        merged = merge_run_caches(data_fp, cfg_fp, benchmark_path)
        if merged:
            print(f"\nAll {N_RUNS} runs complete — merged benchmark cache.")
            print_results(
                merged["hamming_runs"],
                merged["accuracy_runs"],
                np.asarray(merged["last_selected"]) if merged.get("last_selected") else None,
                elapsed=elapsed,
            )
        else:
            done = sum(
                1
                for i in range(N_RUNS)
                if _run_cache_path(data_fp, cfg_fp, i).exists()
            )
            print(f"\nRun {run_idx + 1} done ({done}/{N_RUNS} total cached).")
        return

    t0 = time.perf_counter()
    cached = None if refresh else load_cache(benchmark_path)

    if refresh:
        benchmark_path.unlink(missing_ok=True)
        for i in range(N_RUNS):
            _clear_run_cache(data_fp, cfg_fp, i)

    if not refresh:
        merged = merge_run_caches(data_fp, cfg_fp, benchmark_path)
        if merged:
            cached = merged

    if cached and cached.get("completed_runs", 0) >= N_RUNS:
        print(f"\nUsing cached benchmark ({N_RUNS}/{N_RUNS} runs): {benchmark_path.name}")
        hamming_runs = list(cached["hamming_runs"])
        accuracy_runs = list(cached["accuracy_runs"])
        last_selected = (
            np.asarray(cached["last_selected"], dtype=np.int64)
            if cached.get("last_selected") is not None
            else None
        )
    else:
        print()
        for run in range(N_RUNS):
            run_path = _run_cache_path(data_fp, cfg_fp, run)
            if not refresh and run_path.exists():
                continue
            run_one(
                x,
                y,
                run,
                data_fp,
                cfg_fp,
                refresh=refresh,
                show_d2f_progress=True,
            )
        merged = merge_run_caches(data_fp, cfg_fp, benchmark_path)
        if not merged:
            done = sum(
                1
                for i in range(N_RUNS)
                if _run_cache_path(data_fp, cfg_fp, i).exists()
            )
            raise RuntimeError(f"Benchmark incomplete: {done}/{N_RUNS} runs cached")
        hamming_runs = merged["hamming_runs"]
        accuracy_runs = merged["accuracy_runs"]
        last_selected = (
            np.asarray(merged["last_selected"], dtype=np.int64)
            if merged.get("last_selected") is not None
            else None
        )

    elapsed = time.perf_counter() - t0
    print_results(hamming_runs, accuracy_runs, last_selected, elapsed=elapsed)


def merge_only() -> None:
    data_fp, cfg_fp, benchmark_path = _cache_paths()
    print_benchmark_header()
    t0 = time.perf_counter()
    merged = merge_run_caches(data_fp, cfg_fp, benchmark_path)
    if not merged:
        done = [
            i
            for i in range(N_RUNS)
            if _run_cache_path(data_fp, cfg_fp, i).exists()
        ]
        raise RuntimeError(
            f"Cannot merge: {len(done)}/{N_RUNS} per-run caches found "
            f"(missing runs: {[i for i in range(N_RUNS) if i not in done]})"
        )
    print(f"\nMerged {N_RUNS} runs -> {benchmark_path.name}")
    print_results(
        merged["hamming_runs"],
        merged["accuracy_runs"],
        np.asarray(merged["last_selected"]) if merged.get("last_selected") else None,
        elapsed=time.perf_counter() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="D2F Scene benchmark")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and retrain")
    parser.add_argument(
        "--run",
        type=int,
        metavar="N",
        help=f"Run only split N (0..{N_RUNS - 1}); safe for parallel execution",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge per-run caches and print results (no training)",
    )
    args = parser.parse_args()

    if args.run is not None and (args.run < 0 or args.run >= N_RUNS):
        parser.error(f"--run must be between 0 and {N_RUNS - 1}")

    print("=== Data ===")
    ensure_scene()

    if args.merge_only:
        merge_only()
    else:
        run_benchmark(refresh=args.refresh, run_idx=args.run)
    print("\nDone.")


if __name__ == "__main__":
    main()
