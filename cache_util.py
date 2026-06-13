"""Disk cache for D2F + MLNB training results (invalidates when data or config changes)."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/cache")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def data_fingerprint(*paths: Path) -> str:
  parts: list[str] = []
  for path in paths:
      if path.exists():
          stat = path.stat()
          parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
      else:
          parts.append(f"{path}:missing")
  return _hash_text("|".join(parts))


def config_fingerprint(config: dict[str, Any]) -> str:
    return _hash_text(json.dumps(config, sort_keys=True, default=str))


def cache_file(name: str, data_fp: str, config_fp: str) -> Path:
    return CACHE_DIR / f"{name}_{data_fp}_{config_fp}.pkl"


def load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save_cache(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp.replace(path)
    return path
