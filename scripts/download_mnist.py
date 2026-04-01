#!/usr/bin/env python3
"""
download_mnist.py
Purpose: Download and decompress raw MNIST IDX files for the C++ pipeline.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

BASE_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def download_one(file_name: str, out_dir: Path) -> None:
    gz_path = out_dir / file_name
    raw_path = out_dir / file_name.replace(".gz", "")

    if raw_path.exists():
        print(f"[skip] {raw_path.name} already exists")
        return

    if not gz_path.exists():
        last_error: Exception | None = None
        for base in BASE_URLS:
            url = base + file_name
            try:
                print(f"[download] {url}")
                urllib.request.urlretrieve(url, gz_path)
                last_error = None
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"Failed to download {file_name}: {last_error}")

    print(f"[extract] {gz_path.name} -> {raw_path.name}")
    with gzip.open(gz_path, "rb") as src, raw_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw MNIST IDX files")
    parser.add_argument("--out_dir", type=str, default="./data", help="Output directory for raw IDX files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_name in FILES:
        download_one(file_name, out_dir)

    print("MNIST raw IDX files are ready.")


if __name__ == "__main__":
    main()
