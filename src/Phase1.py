"""Core routines for Phase 1: training models and collecting metrics.

The module orchestrates the learning phase:
1. Load dataset
2. Preprocess (handle missing values, encode categoricals)
3. Extract metadata
4. Train candidate models
5. Measure performance and resource usage
6. Store in knowledge base
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
from codecarbon import EmissionsTracker
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.metadata_extractor import extract_metadata_from_df
from src.preprocessing import preprocess_dataset
from src.config import get_dataset_path, list_datasets, DATASETS


def get_model_pool() -> Dict[str, ClassifierMixin]:
    """Return a dict mapping model name to an uninitialized estimator."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", use_label_encoder=False),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }


def measure_energy_and_train(
    model: ClassifierMixin, X_train, y_train, model_name: str
) -> Tuple[ClassifierMixin, Dict[str, Any]]:
    """Train model and measure energy/CPU time, returning (trained_model, metrics_dict)."""
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem = process.memory_info().rss / (1024 * 1024)
    emissions = 0.0
    try:
        tracker = EmissionsTracker(log_level="error")
        tracker.start()
        
        model.fit(X_train, y_train)
        
        emissions = tracker.stop()
    except Exception as e:
        model.fit(X_train, y_train)
        emissions = 0.0

    # Record end CPU time and memory
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)

    train_time_sec = end_time - start_time
    mem_delta_mb = max(0, end_mem - start_mem)

    co2_kg = emissions if emissions > 0 else 0.0
    energy_kwh = co2_kg / 0.4 if co2_kg > 0 else 0.0

    stats = {
        "train_time_sec": train_time_sec,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "memory_delta_mb": mem_delta_mb,
    }
    return model, stats


def run_dataset_experiment(dataset_key: str, root: Path) -> List[Dict[str, Any]]:
    dataset_path = get_dataset_path(dataset_key, root=root)
    if not Path(dataset_path).exists():
        print(f"Warning: Dataset {dataset_path} not found. Skipping.")
        return []
    
    filename = Path(dataset_path).name
    print(f"\n{'='*60}\nDataset Name: {dataset_key} | File: {filename}\n{'='*60}")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return []

    target_col = DATASETS[dataset_key].get("target_col")
    if target_col is None or target_col not in df.columns:
        print(f"Warning: target column '{target_col}' not found. Using last column.")
        target_col = df.columns[-1]

    try:
        X_train, X_test, y_train, y_test = preprocess_dataset(df, target_col=target_col)
        print(f"Preprocessed: X_train {X_train.shape}, X_test {X_test.shape}")
    except Exception as e:
        print(f"Error preprocessing {dataset_key}: {e}")
        return []

    try:
        metadata = extract_metadata_from_df(X_train)
        print(f"Extracted metadata: n_rows={metadata['n_rows']}, n_features={metadata['n_columns']}")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        metadata = {}

    pool = get_model_pool()
    results: List[Dict[str, Any]] = []
    for name, model in pool.items():
        print(f"  Training {name}...", end=" ", flush=True)
        try:
            trained, stats = measure_energy_and_train(model, X_train.values, y_train.values, name)

            y_pred = trained.predict(X_test)
            accuracy = accuracy_score(y_test.values, y_pred)
            precision = precision_score(y_test.values, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test.values, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test.values, y_pred, average="weighted", zero_division=0)

            print(f"Accuracy: {accuracy:.4f}, Time: {stats['train_time_sec']:.2f}s")

            entry = {
                "dataset_name": dataset_key,
                "dataset_size": len(X_train),
                "n_features": X_train.shape[1],
                "n_rows": metadata.get("n_rows", len(X_train)),
                "n_columns": metadata.get("n_columns", X_train.shape[1]),
                "n_numeric": metadata.get("n_numeric", 0),
                "n_categorical": metadata.get("n_categorical", 0),
                "missing_fraction": metadata.get("missing_fraction", 0),
                "mean_cardinality": metadata.get("mean_cardinality", 0),
                "model_name": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "train_time_sec": stats["train_time_sec"],
                "energy_kwh": stats["energy_kwh"],
                "co2_kg": stats["co2_kg"],
                "memory_delta_mb": stats["memory_delta_mb"],
            }
            results.append(entry)

        except Exception as e:
            print(f"Error training {name}: {e}")

    return results


def run_all(output_csv: Path, root: Path = None) -> None:
    if root is None:
        root = Path.cwd()
    else:
        root = Path(root)

    all_results: List[Dict[str, Any]] = []
    for dataset_key in list_datasets():
        try:
            res = run_dataset_experiment(dataset_key, root=root)
            all_results.extend(res)
        except Exception as ex:
            print(f"error on {dataset_key}: {ex}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv} ({len(df)} rows)\nSummary by dataset:")
        for ds in df["dataset_name"].unique():
            subset = df[df["dataset_name"] == ds]
            best_model = subset.loc[subset["accuracy"].idxmax(), "model_name"]
            best_acc = subset["accuracy"].max()
            print(f"  {ds}: best={best_model} (acc={best_acc:.4f})")
    else:
        print("No results to save.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="knowledge_base.csv", help="CSV file to save results")
    parser.add_argument("--root", default=".", help="Root directory for dataset paths")
    args = parser.parse_args()
    run_all(Path(args.output), Path(args.root))
