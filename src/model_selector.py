"""Model Selector: Constraint-Aware, Explainable Model Selection.

This module implements Phase 2 functionality:
1. Load knowledge base from Phase 1
2. Train a meta-model to predict best model from dataset metadata
3. Apply user constraints (accuracy, energy, carbon)
4. Select and explain the best model choice
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class ModelSelector:
    """Constraint-aware model selection with explainability."""

    def __init__(self, knowledge_base_path: str | Path = "knowledge_base.csv"):
        """Initialize selector by loading knowledge base.

        Parameters
        ----------
        knowledge_base_path : str or Path
            Path to the knowledge_base.csv generated from Phase 1.
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base = None
        self.meta_model = None
        self.model_encoder = None
        self.constraints: Dict[str, Any] = {}

        # Load knowledge base
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")

        self.knowledge_base = pd.read_csv(self.knowledge_base_path)
        print(f"Loaded knowledge base: {len(self.knowledge_base)} entries")

    def set_constraints(
        self,
        min_accuracy: Optional[float] = None,
        max_energy_kwh: Optional[float] = None,
        max_co2_kg: Optional[float] = None,
    ) -> None:
        """Set user constraints for model selection.

        Parameters
        ----------
        min_accuracy : float, optional
            Minimum acceptable accuracy (0.0 to 1.0).
        max_energy_kwh : float, optional
            Maximum acceptable energy in kWh.
        max_co2_kg : float, optional
            Maximum acceptable CO2 emissions in kg.
        """
        self.constraints = {}
        if min_accuracy is not None:
            self.constraints["min_accuracy"] = min_accuracy
        if max_energy_kwh is not None:
            self.constraints["max_energy_kwh"] = max_energy_kwh
        if max_co2_kg is not None:
            self.constraints["max_co2_kg"] = max_co2_kg

        print(f"Constraints set: {self.constraints}")

    def train_meta_model(self) -> None:
        """Train a meta-model to predict best model from dataset metadata.

        The meta-model learns: dataset_metadata → best_model_name
        Uses a RandomForest classifier on metadata features.
        """
        # Identify metadata features and target
        metadata_features = [
            "n_rows",
            "n_columns",
            "n_numeric",
            "n_categorical",
            "missing_fraction",
            "mean_cardinality",
        ]

        kb = self.knowledge_base.copy()
        best_models = (
            kb.groupby("dataset_name")
            .apply(lambda x: x.loc[x["accuracy"].idxmax(), "model_name"])
            .reset_index()
        )
        best_models.columns = ["dataset_name", "best_model"]

        train_data = (
            kb[["dataset_name"] + metadata_features]
            .drop_duplicates(subset=["dataset_name"])
            .merge(best_models, on="dataset_name")
        )

        X = train_data[metadata_features].values
        y = train_data["best_model"].values

        self.model_encoder = LabelEncoder()
        y_encoded = self.model_encoder.fit_transform(y)

        self.meta_model = RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=10
        )
        self.meta_model.fit(X, y_encoded)

        train_acc = accuracy_score(y_encoded, self.meta_model.predict(X))
        print(f"Meta-model trained. Train accuracy: {train_acc:.4f}")

    def save_meta_model(self, path: str | Path) -> None:
        """Save trained meta-model and encoder to disk."""
        p = Path(path)
        with p.open("wb") as f:
            pickle.dump({"meta_model": self.meta_model, "model_encoder": self.model_encoder}, f)
        print(f"Saved meta-model to {p}")

    def load_meta_model(self, path: str | Path) -> None:
        """Load a previously saved meta-model and encoder from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Meta-model file not found: {p}")
        with p.open("rb") as f:
            data = pickle.load(f)
        self.meta_model = data.get("meta_model")
        self.model_encoder = data.get("model_encoder")
        print(f"Loaded meta-model from {p}")

    def predict_model_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Predict best model for given dataset metadata.

        Parameters
        ----------
        metadata : dict
            Dataset metadata dict with keys: n_rows, n_columns, n_numeric, etc.

        Returns
        -------
        model_name : str
            Predicted best model name.
        """
        if self.meta_model is None:
            raise RuntimeError(
                "Meta-model not trained. Call train_meta_model() first."
            )

        metadata_features = [
            "n_rows",
            "n_columns",
            "n_numeric",
            "n_categorical",
            "missing_fraction",
            "mean_cardinality",
        ]
        X = np.array(
            [[metadata.get(feat, 0) for feat in metadata_features]]
        )

        pred_idx = self.meta_model.predict(X)[0]
        return self.model_encoder.inverse_transform([pred_idx])[0]

    def select_best_model(
        self, metadata: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], str]:
        """Select best model given dataset metadata and constraints.

        Parameters
        ----------
        metadata : dict
            Dataset metadata.

        Returns
        -------
        best_model : str
            Selected model name.
        metrics : dict
            Metrics for the selected model (accuracy, energy, co2, etc.).
        explanation : str
            Human-readable explanation of why this model was chosen.
        """
        if self.meta_model is None:
            raise RuntimeError(
                "Meta-model not trained. Call train_meta_model() first."
            )

        predicted_model = self.predict_model_from_metadata(metadata)

        kb = self.knowledge_base.copy()

        candidates = kb.copy()

        if "min_accuracy" in self.constraints:
            candidates = candidates[
                candidates["accuracy"] >= self.constraints["min_accuracy"]
            ]

        if "max_energy_kwh" in self.constraints:
            candidates = candidates[
                candidates["energy_kwh"] <= self.constraints["max_energy_kwh"]
            ]

        if "max_co2_kg" in self.constraints:
            candidates = candidates[
                candidates["co2_kg"] <= self.constraints["max_co2_kg"]
            ]

        if len(candidates) == 0:
            print("Warning: No model satisfies constraints. Returning best overall.")
            candidates = kb.copy()

        preferred = candidates[candidates["model_name"] == predicted_model].copy()
        if len(preferred) > 0:
            energy_max = preferred["energy_kwh"].max()
            co2_max = preferred["co2_kg"].max()
            
            energy_norm = (preferred["energy_kwh"] / energy_max) if energy_max > 0 else 0
            co2_norm = (preferred["co2_kg"] / co2_max) if co2_max > 0 else 0
            
            preferred["goodness"] = (
                preferred["accuracy"] - energy_norm - co2_norm
            )
            
            if preferred["goodness"].isna().all():
                best_row = preferred.iloc[0]
            else:
                best_row = preferred.loc[preferred["goodness"].idxmax()]
        else:
            best_row = candidates.loc[candidates["accuracy"].idxmax()]

        best_model = best_row["model_name"]
        metrics = {
            "accuracy": best_row["accuracy"],
            "energy_kwh": best_row["energy_kwh"],
            "co2_kg": best_row["co2_kg"],
            "train_time_sec": best_row["train_time_sec"],
            "memory_delta_mb": best_row["memory_delta_mb"],
        }

        explanation = self._generate_explanation(
            best_model, predicted_model, metrics, metadata
        )

        return best_model, metrics, explanation

    def _generate_explanation(
        self,
        selected_model: str,
        predicted_model: str,
        metrics: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """Generate human-readable explanation for model selection.

        Parameters
        ----------
        selected_model : str
            The selected model.
        predicted_model : str
            The predicted model from meta-model.
        metrics : dict
            Selected model metrics.
        metadata : dict
            Dataset metadata.

        Returns
        -------
        explanation : str
            Human-readable explanation.
        """
        lines = []
        lines.append(f"Selected Model: {selected_model}")
        lines.append("")

        # Dataset characteristics
        lines.append("Dataset Characteristics:")
        lines.append(f"  - Rows: {int(metadata.get('n_rows', 0))}")
        lines.append(f"  - Features: {int(metadata.get('n_columns', 0))}")
        lines.append(
            f"  - Numeric: {int(metadata.get('n_numeric', 0))}, "
            f"Categorical: {int(metadata.get('n_categorical', 0))}"
        )
        lines.append("")

        # Meta-model insight
        lines.append("Meta-Model Prediction:")
        if selected_model == predicted_model:
            lines.append(
                f"  ✓ {predicted_model} was the top prediction and matched constraints."
            )
        else:
            lines.append(
                f"  {predicted_model} was predicted, but {selected_model} "
                f"better balances accuracy with resource constraints."
            )
        lines.append("")

        # Performance justification
        lines.append("Performance Metrics:")
        lines.append(f"  - Accuracy: {metrics['accuracy']:.4f}")
        lines.append(f"  - Energy: {metrics['energy_kwh']:.6f} kWh")
        lines.append(f"  - CO₂ Emissions: {metrics['co2_kg']:.6f} kg")
        lines.append(f"  - Training Time: {metrics['train_time_sec']:.2f} sec")
        lines.append("")

        # Constraint satisfaction
        if self.constraints:
            lines.append("Constraint Satisfaction:")
            for key, value in self.constraints.items():
                if key == "min_accuracy":
                    satisfy = "✓" if metrics["accuracy"] >= value else "✗"
                    lines.append(
                        f"  {satisfy} Min Accuracy: {value:.4f} "
                        f"(achieved: {metrics['accuracy']:.4f})"
                    )
                elif key == "max_energy_kwh":
                    satisfy = "✓" if metrics["energy_kwh"] <= value else "✗"
                    lines.append(
                        f"  {satisfy} Max Energy: {value:.6f} kWh "
                        f"(used: {metrics['energy_kwh']:.6f})"
                    )
                elif key == "max_co2_kg":
                    satisfy = "✓" if metrics["co2_kg"] <= value else "✗"
                    lines.append(
                        f"  {satisfy} Max CO₂: {value:.6f} kg "
                        f"(emitted: {metrics['co2_kg']:.6f})"
                    )

        return "\n".join(lines)

    def save_meta_model(self, path: str | Path = "meta_model.pkl") -> None:
        """Save trained meta-model to pickle file.

        Parameters
        ----------
        path : str or Path
            Output path for meta-model pickle.
        """
        if self.meta_model is None:
            raise RuntimeError("No meta-model to save.")

        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.meta_model, "encoder": self.model_encoder}, f
            )
        print(f"Meta-model saved to {path}")

    def load_meta_model(self, path: str | Path = "meta_model.pkl") -> None:
        """Load trained meta-model from pickle file.

        Parameters
        ----------
        path : str or Path
            Path to meta-model pickle file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Meta-model not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.meta_model = data["model"]
            self.model_encoder = data["encoder"]
        print(f"Meta-model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    selector = ModelSelector("knowledge_base.csv")

    # Train meta-model
    selector.train_meta_model()

    # Save for later reuse
    selector.save_meta_model("meta_model.pkl")

    # Set constraints: min accuracy 0.85, max energy 0.00001 kWh
    selector.set_constraints(min_accuracy=0.85, max_energy_kwh=0.00001)

    # Simulate a new dataset metadata
    sample_metadata = {
        "n_rows": 10000,
        "n_columns": 20,
        "n_numeric": 18,
        "n_categorical": 2,
        "missing_fraction": 0.05,
        "mean_cardinality": 500.0,
    }

    # Select best model
    best_model, metrics, explanation = selector.select_best_model(sample_metadata)

    print("\n" + "=" * 60)
    print(explanation)
    print("=" * 60)
