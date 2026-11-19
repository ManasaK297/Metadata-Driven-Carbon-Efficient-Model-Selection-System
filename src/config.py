"""Configuration for datasets in the learning phase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_DATASETS: Dict[str, Dict[str, Any]] = {
    "Dataset1": {
        "path": "Data/Bank Marketing Dataset/bank.csv",
        "target_col": "deposit",
        "description": "Bank marketing campaign response prediction",
    },
    "Dataset2": {
        "path": "Data/Credit Card Default Dataset/UCI_Credit_Card.csv",
        "target_col": "default.payment.next.month",
        "description": "Credit card default prediction",
    },
    "Dataset3": {
        "path": "Data/Financial Fraud/cards_data.csv",
        "target_col": None,
        "description": "Card-level financial data",
    },
    "Dataset4": {
        "path": "Data/Financial Fraud/users_data.csv",
        "target_col": None,
        "description": "User-level financial data",
    },
    "Dataset5": {
        "path": "Data/Financial Distress Prediction/Financial Distress.csv",
        "target_col": "Financial Distress",
        "description": "Financial distress prediction",
    },
    "Dataset6": {
        "path": "Data/Loan Prediction/train_u6lujuX_CVtuZ9i.csv",
        "target_col": "Loan_Status",
        "description": "Loan approval prediction",
    },
    "Dataset7": {
        "path": "Data/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "target_col": "Churn",
        "description": "Telco customer churn prediction",
    },
    "Dataset8": {
        "path": "Data/German Credit Risk/german_credit_data.csv",
        "target_col": "Risk",
        "description": "German credit risk assessment",
    },
    "Dataset9": {
        "path": "Data/Loan Prediction/test_Y3wMUE5_7gLdaTN.csv",
        "target_col": None,
        "description": "Loan prediction test set",
    },
}


def load_datasets(config_path: str | Path = "config.json") -> Dict[str, Dict[str, Any]]:
    """Load datasets from JSON config file, or return defaults if not found."""
    config_path = Path(config_path)
    project_root = Path(__file__).parents[1]

    # Start with either the user-provided config or defaults
    if config_path.exists():
        with open(config_path, "r") as f:
            base = json.load(f)
    else:
        base = dict(DEFAULT_DATASETS)

    # Remove entries from base whose files no longer exist on disk
    cleaned_base: Dict[str, Dict[str, Any]] = {}
    for k, v in base.items():
        rel = v.get("path")
        if not rel:
            continue
        p = project_root / Path(rel)
        if p.exists():
            cleaned_base[k] = v

    base = cleaned_base

    # Auto-discover any CSVs under the Data/ folder that are not already listed
    discovered: Dict[str, Dict[str, Any]] = {}
    data_root = project_root / "Data"
    if data_root.exists():
        for p in data_root.rglob("*.csv"):
            try:
                rel = p.relative_to(project_root).as_posix()
            except Exception:
                rel = p.as_posix()

            # create a stable key based on the path under Data/
            try:
                key = p.relative_to(data_root).with_suffix("").as_posix().replace("/", "_")
            except Exception:
                key = p.stem

            if key in base:
                continue

            discovered[key] = {
                "path": rel,
                "target_col": None,
                "description": f"Auto-discovered dataset {rel}",
            }

    # Merge base (user/default) with discovered datasets
    merged = dict(base)
    merged.update(discovered)
    return merged


DATASETS = load_datasets()


def get_dataset_path(dataset_key: str, root: str | Path = ".") -> str:
    """Return the full path to a dataset file."""
    root = Path(root)
    rel_path = DATASETS[dataset_key]["path"]
    return str(root / rel_path)


def list_datasets() -> list[str]:
    """Return list of available dataset keys."""
    return list(DATASETS.keys())


if __name__ == "__main__":
    for key in list_datasets():
        info = DATASETS[key]
        print(f"{key}: {info['description']} (target: {info['target_col']})")
