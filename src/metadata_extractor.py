"""Utility for extracting dataset metadata features.

This module defines functions to compute a standardized set of
characteristics for a tabular dataset given as a CSV file or DataFrame.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def extract_metadata_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract metadata: n_rows, n_columns, n_numeric, n_categorical, missing_fraction, mean_cardinality."""

    n_rows, n_columns = df.shape
    dtypes = df.dtypes

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_per_column = df.isna().mean()
    cardinalities = df.nunique(dropna=True)

    metadata: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "missing_fraction": float(df.isna().mean().mean()),
        "mean_cardinality": float(cardinalities.mean()) if not cardinalities.empty else 0.0,
        "min_cardinality": int(cardinalities.min()) if not cardinalities.empty else 0,
        "max_cardinality": int(cardinalities.max()) if not cardinalities.empty else 0,
    }

    metadata["high_missing_cols"] = int((missing_per_column > 0.5).sum())

    metadata["many_levels_cols"] = int((cardinalities > 10).sum())

    return metadata


def extract_metadata_from_csv(path: str, **read_csv_kwargs) -> Dict[str, Any]:
    """Load a CSV file and extract metadata.

    Parameters
    ----------
    path : str
        Path to the CSV dataset.
    read_csv_kwargs : kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    metadata : dict
        Metadata dictionary returned by :func:`extract_metadata_from_df`.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, **read_csv_kwargs)
    return extract_metadata_from_df(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract metadata features from CSV datasets."
    )
    parser.add_argument("files", nargs="+", help="Paths to CSV files to inspect")
    args = parser.parse_args()

    for f in args.files:
        try:
            meta = extract_metadata_from_csv(f)
            print(f"{os.path.basename(f)} -> {meta}\n")
        except Exception as ex:
            print(f"error processing {f}: {ex}\n")
