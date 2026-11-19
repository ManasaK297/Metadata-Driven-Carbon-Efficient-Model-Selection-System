"""AI Explanation module using Google Gemini with an offline fallback."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import requests


class LLMExplainer(ABC):
    """Abstract base class for LLM explainers."""

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if the LLM service is available."""
        pass

    @abstractmethod
    def generate_explanation(
        self,
        selected_model: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        predicted_model: str,
    ) -> str:
        """Generate explanation for model selection."""
        pass

    def _build_prompt(
        self,
        selected_model: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        predicted_model: str,
    ) -> str:
        """Build prompt for the LLM."""
        return f"""
You are explaining a metadata-driven, constraint-aware machine learning project.
Explain in a concise but professional way why {selected_model} was selected for this dataset.

Dataset Characteristics:
- Rows: {metadata.get('n_rows', 'unknown'):,}
- Features: {metadata.get('n_columns', 'unknown')}
- Numeric Features: {metadata.get('n_numeric', 'unknown')}
- Categorical Features: {metadata.get('n_categorical', 'unknown')}
- Missing Data Fraction: {metadata.get('missing_fraction', 0):.2%}
- Mean Cardinality: {metadata.get('mean_cardinality', 0):.0f}

Model Performance:
- Test Accuracy: {metrics.get('accuracy', 0):.2%}
- Training Time: {metrics.get('train_time_sec', 0):.2f} seconds
- Energy Used: {metrics.get('energy_kwh', 0):.6f} kWh
- CO₂ Emissions: {metrics.get('co2_kg', 0):.6f} kg

Meta-Model Prediction: {predicted_model}
Selected Model: {selected_model}

Write a polished 4-sentence technical explanation for a viva/demo audience. Mention:
1. Why {selected_model} matches the dataset profile
2. How the dataset size, feature mix, and complexity influenced the decision
3. Why this choice balances accuracy with efficiency and sustainability
4. Why this is suitable for metadata-based model selection rather than random model choice

Keep it concise, clear, and project-specific."""

    def _fallback_explanation(
        self,
        selected_model: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> str:
        """Fallback explanation used when Gemini is unavailable."""
        dataset_size = "large" if metadata.get("n_rows", 0) > 100000 else "medium"
        feature_count = metadata.get("n_columns", 0)

        explanations = {
            "LogisticRegression": f"{selected_model} was chosen because the dataset is structured and moderately sized, making a linear model efficient and stable while maintaining strong interpretability.",
            "DecisionTree": f"{selected_model} was selected because the dataset has a manageable size and feature set, where a tree-based model can capture non-linear patterns with low computational cost.",
            "RandomForest": f"{selected_model} was selected because the dataset profile favors ensemble learning: it handles non-linear relationships well, reduces variance, and maintains a strong accuracy-efficiency tradeoff.",
            "XGBoost": f"{selected_model} was selected because the dataset has enough complexity to benefit from gradient boosting, while still meeting the project’s sustainability constraints with high predictive performance.",
        }

        base_explanation = explanations.get(
            selected_model,
            f"{selected_model} was selected based on the dataset’s structural properties and the project’s constraint-aware model selection strategy.",
        )

        return f"""{base_explanation} The dataset contains {metadata.get('n_rows', 0):,} rows and {feature_count} features, which makes this model suitable for the observed complexity and feature distribution. It delivers {metrics.get('accuracy', 0):.2%} accuracy with {metrics.get('train_time_sec', 0):.2f}s training time while keeping emissions low at {metrics.get('co2_kg', 0):.6f} kg CO₂. This demonstrates the core idea of the project: choosing the best model by metadata, constraints, and efficiency rather than training every model blindly."""


class GoogleGeminiExplainer(LLMExplainer):
    """Generate AI-powered explanations using Google Gemini API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = "gemini-pro"
        self.api_endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-pro:generateContent"
        )

    def check_connection(self) -> bool:
        """Check if the Google Gemini API is reachable."""
        if not self.api_key:
            return False
        try:
            response = requests.post(
                self.api_endpoint,
                json={"contents": [{"parts": [{"text": "test"}]}]},
                params={"key": self.api_key},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def generate_explanation(
        self,
        selected_model: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        predicted_model: str,
    ) -> str:
        """Generate a Gemini explanation or fall back to local text."""
        if not self.api_key:
            return self._fallback_explanation(selected_model, metadata, metrics)

        prompt = self._build_prompt(selected_model, metadata, metrics, predicted_model)

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 500,
                    },
                },
                params={"key": self.api_key},
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    pass
        except Exception as e:
            print(f"Error calling Google Gemini: {e}")

        return self._fallback_explanation(selected_model, metadata, metrics)


if __name__ == "__main__":
    explainer = GoogleGeminiExplainer()

    if explainer.check_connection():
        print("✅ Gemini is reachable.")
        test_metadata = {
            "n_rows": 100000,
            "n_columns": 20,
            "n_numeric": 18,
            "n_categorical": 2,
            "missing_fraction": 0.01,
            "mean_cardinality": 120,
        }
        test_metrics = {
            "accuracy": 0.96,
            "train_time_sec": 3.2,
            "energy_kwh": 0.00002,
            "co2_kg": 0.000004,
        }
        explanation = explainer.generate_explanation(
            "RandomForest", test_metadata, test_metrics, "RandomForest"
        )
        print("\nGenerated Explanation:")
        print(explanation)
    else:
        print("❌ Gemini is not available. Using fallback explanation only.")
        fallback = explainer._fallback_explanation(
            "RandomForest",
            {"n_rows": 100000, "n_columns": 20},
            {"accuracy": 0.96, "train_time_sec": 3.2, "co2_kg": 0.000004},
        )
        print(fallback)
