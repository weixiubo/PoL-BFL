#!/usr/bin/env python3
"""Prepare FEMNIST writer-partition data from the public Hugging Face parquet.

The project loader expects LEAF-style JSON shards with users/user_data fields.
This script downloads the Flower Labs FEMNIST parquet mirror and converts it
into deterministic per-writer train/test JSON shards under data/FEMNIST.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_URL = "https://huggingface.co/datasets/flwrlabs/femnist/resolve/main/data/train-00000-of-00001.parquet"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def _image_to_flat(image_value: Any) -> List[float]:
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image = Image.open(io.BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            image = Image.open(image_value["path"])
        else:
            raise ValueError("Unsupported HF image dict: missing bytes/path")
    elif isinstance(image_value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_value))
    elif isinstance(image_value, Image.Image):
        image = image_value
    else:
        arr = np.asarray(image_value)
        if arr.size == 28 * 28:
            return (arr.astype(np.float32).reshape(-1) / (255.0 if arr.max() > 1 else 1.0)).tolist()
        raise ValueError(f"Unsupported image value type: {type(image_value)!r}")

    image = image.convert("L").resize((28, 28))
    arr = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return arr.tolist()


def _label_to_int(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    digits = "0123456789"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lowercase = "abcdefghijklmnopqrstuvwxyz"
    labels = {ch: i for i, ch in enumerate(digits + uppercase + lowercase)}
    if text in labels:
        return labels[text]
    raise ValueError(f"Cannot map FEMNIST character label to int: {value!r}")


def _load_rows(parquet_path: Path) -> Iterable[Tuple[str, List[float], int]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("prepare_femnist_hf.py requires pandas with parquet support") from exc

    df = pd.read_parquet(parquet_path)
    required = {"image", "writer_id", "character"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FEMNIST parquet missing required columns: {sorted(missing)}")
    for row in df.itertuples(index=False):
        data = row._asdict()
        yield str(data["writer_id"]), _image_to_flat(data["image"]), _label_to_int(data["character"])


def _write_shards(users: Dict[str, Dict[str, List]], out_dir: Path, shard_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(users.items(), key=lambda item: item[0])
    total = len(items)
    num_shards = max(1, math.ceil(total / shard_size))
    for shard_idx in range(num_shards):
        chunk = items[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        payload = {
            "users": [user for user, _ in chunk],
            "num_samples": [len(data["y"]) for _, data in chunk],
            "user_data": {user: data for user, data in chunk},
        }
        path = out_dir / f"user_data_{shard_idx}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def convert(parquet_path: Path, output_root: Path, train_fraction: float, shard_size: int) -> Dict[str, Any]:
    train_users: Dict[str, Dict[str, List]] = defaultdict(lambda: {"x": [], "y": []})
    test_users: Dict[str, Dict[str, List]] = defaultdict(lambda: {"x": [], "y": []})
    per_user_count: Dict[str, int] = defaultdict(int)

    for writer_id, flat, label in _load_rows(parquet_path):
        count = per_user_count[writer_id]
        per_user_count[writer_id] += 1
        # Deterministic per-writer sample split. This preserves natural writer
        # partitions while creating a test split for the single-split HF mirror.
        bucket = train_users if (count % 100) < int(train_fraction * 100) else test_users
        bucket[writer_id]["x"].append(flat)
        bucket[writer_id]["y"].append(label)

    _write_shards(train_users, output_root / "train", shard_size)
    _write_shards(test_users, output_root / "test", shard_size)
    return {
        "parquet_path": str(parquet_path),
        "output_root": str(output_root),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "train_samples": sum(len(data["y"]) for data in train_users.values()),
        "test_samples": sum(len(data["y"]) for data in test_users.values()),
        "train_fraction": train_fraction,
        "shard_size": shard_size,
        "source_url": DEFAULT_URL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/FEMNIST"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    parquet_path = data_root / "hf" / "train-00000-of-00001.parquet"
    if args.force and parquet_path.exists():
        parquet_path.unlink()
    _download(args.url, parquet_path)
    summary = convert(parquet_path, data_root, float(args.train_fraction), int(args.shard_size))
    summary_path = data_root / "prepare_femnist_hf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
