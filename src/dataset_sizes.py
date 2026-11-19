"""Print dataset key and CSV file size (bytes) for each configured dataset.

Usage:
    python src/dataset_sizes.py

Output format:
    <dataset_key>: <bytes>
"""

from __future__ import annotations

from pathlib import Path
import argparse

from src.config import DATASETS, get_dataset_path


def main() -> int:
    project_root = Path(__file__).parents[1]

    for key, meta in DATASETS.items():
        rel = meta.get("path")
        if not rel:
            print(f"{key}: no path configured")
            continue

        path = Path(get_dataset_path(key, root=project_root))
        if not path.exists():
            print(f"{key}: file not found")
            continue

        size_bytes = path.stat().st_size
        print(f"{key}: {size_bytes}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print dataset CSV sizes (bytes)")
    _ = parser.parse_args()
    raise SystemExit(main())
