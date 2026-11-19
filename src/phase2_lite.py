"""Phase 2 Lite: Prediction Phase without energy tracking (codecarbon-free).

Simplified version for testing model selection without measuring energy/carbon.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import accuracy_score

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.metadata_extractor import extract_metadata_from_df
from src.preprocessing import preprocess_dataset
from src.model_selector import ModelSelector

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

def get_model_pool():
    """Return dict of model estimators."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }


def run_phase2_lite(
    dataset_path: str | Path,
    target_col: str,
    knowledge_base_path: str | Path = "knowledge_base.csv",
    meta_model_path: str | Path = "meta_model.pkl",
    min_accuracy: float | None = None,
    max_energy_kwh: float | None = None,
    max_co2_kg: float | None = None,
) -> Dict[str, Any]:
    
    print("\n" + "=" * 70)
    print("PHASE 2 LITE: MODEL SELECTION & TRAINING (No Energy Tracking)")
    print("=" * 70)

    # Step 1: Load dataset
    print(f"\n[1/4] Loading dataset from {dataset_path}")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    print(f"\n[2/4] Preprocessing data and handling missing values")
    X_train, X_test, y_train, y_test = preprocess_dataset(df, target_col=target_col)
    print(f"  Preprocessed: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

    print(f"\n[3/4] Extracting dataset metadata")
    metadata = extract_metadata_from_df(X_train)
    print(f"  Metadata extracted:")
    print(f"    - Rows: {metadata['n_rows']}, Features: {metadata['n_columns']}")
    print(f"    - Numeric: {metadata['n_numeric']}, Categorical: {metadata['n_categorical']}")
    print(f"    - Missing fraction: {metadata['missing_fraction']:.4f}")

    print(f"\n[4/4] Selecting and training best model")
    selector = ModelSelector(knowledge_base_path)

    try:
        selector.load_meta_model(meta_model_path)
        print(f"  Meta-model loaded from {meta_model_path}")
    except FileNotFoundError:
        print(f"  Meta-model not found. Training new one...")
        selector.train_meta_model()
        selector.save_meta_model(meta_model_path)
        print(f"  Meta-model trained and saved")

    selector.set_constraints(
        min_accuracy=min_accuracy,
        max_energy_kwh=max_energy_kwh,
        max_co2_kg=max_co2_kg,
    )

    selected_model, metrics, explanation = selector.select_best_model(metadata)
    print(f"  Selected model: {selected_model}")

    print(f"  Training {selected_model}...", end=" ", flush=True)
    model_pool = get_model_pool()
    model = model_pool[selected_model]
    
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    test_accuracy = accuracy_score(y_test.values, y_pred)
    print(f"Done!")
    print(f"  Test accuracy: {test_accuracy:.4f}")

    results = {
        "selected_model": selected_model,
        "metadata": metadata,
        "test_accuracy": test_accuracy,
        "knowledge_base_metrics": metrics,
        "explanation": explanation,
    }

    print("\n" + "=" * 70)
    print("PHASE 2 LITE RESULTS")
    print("=" * 70)
    print(explanation)
    print("=" * 70)
    print(f"\nTest Accuracy on New Dataset: {test_accuracy:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute Phase 2 Lite: Model selection without energy tracking."
    )
    parser.add_argument("dataset", help="Path to new dataset CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--knowledge-base",
        default="knowledge_base.csv",
        help="Path to knowledge_base.csv",
    )
    parser.add_argument(
        "--meta-model", default="meta_model.pkl", help="Path to meta_model.pkl"
    )
    parser.add_argument("--min-accuracy", type=float, help="Min accuracy constraint")
    parser.add_argument("--max-energy", type=float, help="Max energy (kWh) constraint")
    parser.add_argument("--max-co2", type=float, help="Max CO2 (kg) constraint")

    args = parser.parse_args()

    results = run_phase2_lite(
        args.dataset,
        args.target,
        knowledge_base_path=args.knowledge_base,
        meta_model_path=args.meta_model,
        min_accuracy=args.min_accuracy,
        max_energy_kwh=args.max_energy,
        max_co2_kg=args.max_co2,
    )
