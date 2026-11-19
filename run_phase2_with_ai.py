#!/usr/bin/env python3
"""Run Phase 2 with an optional AI explanation summary."""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ai_explainer import GoogleGeminiExplainer
from src.phase2_lite import run_phase2_lite


def resolve_dataset_path(path_str: str) -> str:
    """Resolve a dataset path relative to the project directory."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        return str(candidate)
    return str((project_root / candidate).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 with optional AI explanation."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="Data/Bank Marketing Dataset/bank.csv",
        help="Path to the dataset CSV file.",
    )
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--knowledge-base",
        default="knowledge_base.csv",
        help="Path to the knowledge base CSV",
    )
    parser.add_argument(
        "--meta-model",
        default="meta_model.pkl",
        help="Path to the trained meta-model file",
    )
    parser.add_argument("--min-accuracy", type=float, help="Minimum acceptable accuracy")
    parser.add_argument("--max-energy", type=float, help="Maximum energy used (kWh)")
    parser.add_argument("--max-co2", type=float, help="Maximum CO2 emissions (kg)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    knowledge_base = resolve_dataset_path(args.knowledge_base)
    meta_model = resolve_dataset_path(args.meta_model)

    print("\n" + "=" * 60)
    print("Running Phase 2 with AI Explanations")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Target: {args.target}")
    print(f"Constraints: min_accuracy={args.min_accuracy}, max_energy={args.max_energy}, max_co2={args.max_co2}")

    results = run_phase2_lite(
        dataset_path=dataset_path,
        target_col=args.target,
        knowledge_base_path=knowledge_base,
        meta_model_path=meta_model,
        min_accuracy=args.min_accuracy,
        max_energy_kwh=args.max_energy,
        max_co2_kg=args.max_co2,
    )

    print(f"\nModel Selected: {results['selected_model']}")
    print(f"Test Accuracy: {results['test_accuracy']:.2%}")

    kb_metrics = results['knowledge_base_metrics']
    if isinstance(kb_metrics.get('train_time_sec'), (int, float)):
        print(f"Training Time: {kb_metrics['train_time_sec']:.2f}s")
    else:
        print("Training Time: N/A")
    print(f"CO2 Emissions: {kb_metrics.get('co2_kg', 0):.6f} kg")

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("\nGenerating AI Explanation...")
        explainer = GoogleGeminiExplainer(api_key=api_key)
        explanation = explainer.generate_explanation(
            results['selected_model'],
            results['metadata'],
            results['knowledge_base_metrics'],
            results['selected_model'],
        )
    else:
        print("\nGOOGLE_API_KEY not set. Using offline fallback explanation.")
        explanation = GoogleGeminiExplainer()._fallback_explanation(
            results['selected_model'],
            results['metadata'],
            results['knowledge_base_metrics'],
        )

    print("\n" + "=" * 60)
    print("AI EXPLANATION")
    print("=" * 60)
    print(explanation)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
