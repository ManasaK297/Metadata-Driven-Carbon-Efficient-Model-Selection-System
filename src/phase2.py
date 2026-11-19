"""Phase 2: Prediction Phase - Select and train best model for a new dataset.

This module implements the prediction phase:
1. User provides a new dataset
2. System extracts metadata features
3. User specifies constraints (accuracy, energy, carbon)
4. Meta-model predicts suitable model
5. Only that model is trained
6. Results and explanation are returned
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

# Ensure imports work
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.metadata_extractor import extract_metadata_from_df
from src.preprocessing import preprocess_dataset
from src.model_selector import ModelSelector
from src.Phase1 import measure_energy_and_train, get_model_pool


def run_phase2(
    dataset_path: str | Path,
    target_col: str,
    knowledge_base_path: str | Path = "knowledge_base.csv",
    meta_model_path: str | Path = "meta_model.pkl",
    min_accuracy: float | None = None,
    max_energy_kwh: float | None = None,
    max_co2_kg: float | None = None,
) -> Dict[str, Any]:
    """Execute Phase 2: Select and train best model on new dataset.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the new dataset CSV file.
    target_col : str
        Name of the target column in the dataset.
    knowledge_base_path : str or Path
        Path to knowledge_base.csv from Phase 1.
    meta_model_path : str or Path
        Path to saved meta_model.pkl.
    min_accuracy : float, optional
        Minimum acceptable accuracy constraint.
    max_energy_kwh : float, optional
        Maximum acceptable energy constraint.
    max_co2_kg : float, optional
        Maximum acceptable CO2 constraint.

    Returns
    -------
    results : dict
        Dictionary containing:
        - selected_model: str
        - metadata: dict
        - metrics: dict (accuracy, energy, co2, time, memory)
        - explanation: str
        - preprocessed_data: tuple (X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL SELECTION & TRAINING")
    print("=" * 70)

    # Step 1: Load dataset
    print(f"\n[1/5] Loading dataset from {dataset_path}")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Preprocess
    print(f"\n[2/5] Preprocessing data and handling missing values")
    X_train, X_test, y_train, y_test = preprocess_dataset(df, target_col=target_col)
    print(f"  ✓ Preprocessed: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

    # Step 3: Extract metadata
    print(f"\n[3/5] Extracting dataset metadata")
    metadata = extract_metadata_from_df(X_train)
    print(f"  ✓ Metadata extracted:")
    print(f"    - Rows: {metadata['n_rows']}, Features: {metadata['n_columns']}")
    print(f"    - Numeric: {metadata['n_numeric']}, Categorical: {metadata['n_categorical']}")
    print(f"    - Missing fraction: {metadata['missing_fraction']:.4f}")

    # Step 4: Select model using ModelSelector
    print(f"\n[4/5] Selecting best model using constraints")
    selector = ModelSelector(knowledge_base_path)

    # Load saved meta-model
    try:
        selector.load_meta_model(meta_model_path)
        print(f"  ✓ Meta-model loaded from {meta_model_path}")
    except FileNotFoundError:
        print(f"  ⚠ Meta-model not found. Training new one...")
        selector.train_meta_model()
        selector.save_meta_model(meta_model_path)
        print(f"  ✓ Meta-model trained and saved")

    # Set constraints
    selector.set_constraints(
        min_accuracy=min_accuracy,
        max_energy_kwh=max_energy_kwh,
        max_co2_kg=max_co2_kg,
    )

    # Get model selection
    selected_model, metrics, explanation = selector.select_best_model(metadata)
    print(f"  ✓ Selected model: {selected_model}")

    # Step 5: Train selected model
    print(f"\n[5/5] Training selected model: {selected_model}")
    model_pool = get_model_pool()
    model = model_pool[selected_model]

    trained_model, train_stats = measure_energy_and_train(
        model, X_train.values, y_train.values, selected_model
    )

    # Evaluate on test set
    y_pred = trained_model.predict(X_test.values)
    from sklearn.metrics import accuracy_score

    test_accuracy = accuracy_score(y_test.values, y_pred)
    print(f"  ✓ Model trained. Test accuracy: {test_accuracy:.4f}")

    # Compile results
    results = {
        "selected_model": selected_model,
        "metadata": metadata,
        "training_metrics": {
            "accuracy": test_accuracy,
            "energy_kwh": train_stats["energy_kwh"],
            "co2_kg": train_stats["co2_kg"],
            "train_time_sec": train_stats["train_time_sec"],
            "memory_delta_mb": train_stats["memory_delta_mb"],
        },
        "explanation": explanation,
        "preprocessed_data": (X_train, X_test, y_train, y_test),
        "trained_model": trained_model,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(explanation)
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute Phase 2: Model selection and training on new dataset."
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

    results = run_phase2(
        args.dataset,
        args.target,
        knowledge_base_path=args.knowledge_base,
        meta_model_path=args.meta_model,
        min_accuracy=args.min_accuracy,
        max_energy_kwh=args.max_energy,
        max_co2_kg=args.max_co2,
    )
