"""
D2F: Mutual Information-based Multi-label Feature Selection (Lee & Kim, ESWA 2015).

Direct Python port of reference/D2F-main/program/d2f.m
Input: integer categorical feature matrix X, binary label matrix Y
Output: indices of selected features (0-based)
"""

from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

from cache_util import load_cache, save_cache

T = TypeVar("T")


def _progress(iterable: Iterable[T], desc: str, *, enabled: bool = True) -> Iterable[T]:
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, unit="step", leave=True)


def n_entropy(vector: np.ndarray) -> float:
    """Entropy H(V) for categorical column(s); matches MATLAB unique(...,'rows')."""
    arr = np.atleast_2d(np.asarray(vector))
    _, inverse = np.unique(arr, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=inverse.max() + 1).astype(float)
    probs = counts[counts > 0] / arr.shape[0]
    return float(-np.sum(probs * np.log2(probs)))


def d2f(
    data: np.ndarray,
    target: np.ndarray,
    number: int,
    *,
    progress: bool = True,
    checkpoint: Path | None = None,
) -> np.ndarray:
    """
    Greedy D2F feature selection using interaction information.

    Parameters
    ----------
    data : (n_samples, n_features) int — discretized/categorical features
    target : (n_samples, n_labels) int — multi-label matrix (0/1)
    number : how many features to select

    Returns
    -------
    selected : (number,) int — 0-based feature indices, best first
    """
    data = np.asarray(data, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    n_feat = data.shape[1]
    n_lab = target.shape[1]
    number = min(number, n_feat)

    def _save_checkpoint(
        f_ent_arr: np.ndarray,
        ff_ent_arr: np.ndarray,
        fl_ent_arr: np.ndarray,
        scr_arr: np.ndarray,
        selected_list: list[int],
    ) -> None:
        if checkpoint is None:
            return
        save_cache(
            checkpoint,
            {
                "n_feat": n_feat,
                "n_lab": n_lab,
                "number": number,
                "f_ent": f_ent_arr,
                "ff_ent": ff_ent_arr,
                "fl_ent": fl_ent_arr,
                "scr": scr_arr,
                "selected": selected_list,
            },
        )

    f_ent: np.ndarray | None = None
    ff_ent: np.ndarray | None = None
    fl_ent: np.ndarray | None = None
    scr: np.ndarray | None = None
    selected: list[int] = []

    if checkpoint is not None:
        state = load_cache(checkpoint)
        if (
            state
            and state.get("n_feat") == n_feat
            and state.get("n_lab") == n_lab
            and state.get("number") == number
        ):
            f_ent = np.asarray(state["f_ent"])
            ff_ent = np.asarray(state["ff_ent"])
            fl_ent = np.asarray(state["fl_ent"])
            scr = np.asarray(state["scr"], dtype=float)
            selected = list(state["selected"])
            if progress and selected:
                print(f"D2F: resumed checkpoint ({len(selected)}/{number} features)")

    if f_ent is None:
        f_ent = np.zeros(n_feat)
        ff_ent = np.zeros((n_feat, n_feat))
        fl_ent = np.zeros((n_feat, n_lab))

        for k in _progress(range(n_feat), "D2F precompute", enabled=progress):
            f_ent[k] = n_entropy(data[:, k])
            ff_ent[k, k] = f_ent[k]
            for m in range(k + 1, n_feat):
                h = n_entropy(np.column_stack([data[:, k], data[:, m]]))
                ff_ent[k, m] = ff_ent[m, k] = h
            for m in range(n_lab):
                fl_ent[k, m] = n_entropy(np.column_stack([data[:, k], target[:, m]]))

        scr = np.array([f_ent[k] - fl_ent[k, m] for k in range(n_feat) for m in range(n_lab)])
        scr = scr.reshape(n_feat, n_lab).sum(axis=1)

    assert f_ent is not None and ff_ent is not None and fl_ent is not None and scr is not None

    if len(selected) >= number:
        return np.array(selected[:number], dtype=np.int64)

    if not selected:
        first = int(np.argmax(scr))
        selected.append(first)
        scr[first] = -np.inf
        _save_checkpoint(f_ent, ff_ent, fl_ent, scr, selected)

    for _ in _progress(range(len(selected), number), "D2F select", enabled=progress):
        for m in range(n_feat):
            if np.isinf(scr[m]):
                continue
            for n in selected:
                for a in range(n_lab):
                    triple = np.column_stack([data[:, m], data[:, n], target[:, a]])
                    scr[m] -= (
                        f_ent[m] + f_ent[n] - ff_ent[m, n] - fl_ent[m, a] + n_entropy(triple)
                    )
        nxt = int(np.argmax(scr))
        selected.append(nxt)
        scr[nxt] = -np.inf
        _save_checkpoint(f_ent, ff_ent, fl_ent, scr, selected)

    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink(missing_ok=True)

    return np.array(selected, dtype=np.int64)
