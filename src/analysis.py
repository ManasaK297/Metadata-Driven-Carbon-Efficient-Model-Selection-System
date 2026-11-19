"""Comparative analysis utilities for the knowledge base.

Generates plots comparing models across accuracy, training time, energy, and CO2.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_comparative_plots(kb_path: str | Path = "knowledge_base.csv", out_dir: str | Path = "outputs"):
    kb_path = Path(kb_path)
    out_dir = Path(out_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(kb_path)

    # Aggregate by model
    agg_dict = {
        'accuracy_mean': ('accuracy', 'mean'),
        'train_time_mean': ('train_time_sec', 'mean'),
        'energy_mean': ('energy_kwh', 'mean'),
        'co2_mean': ('co2_kg', 'mean'),
    }
    if 'f1_score' in df.columns:
        agg_dict['f1_mean'] = ('f1_score', 'mean')

    agg = df.groupby('model_name').agg(**agg_dict).reset_index()

    sns.set(style="whitegrid")

    plots = [
        ("accuracy_mean", "Average Accuracy", "accuracy_vs_model.png"),
        ("train_time_mean", "Average Training Time (s)", "train_time_vs_model.png"),
        ("energy_mean", "Average Energy (kWh)", "energy_vs_model.png"),
        ("co2_mean", "Average CO2 (kg)", "co2_vs_model.png"),
    ]
    if 'f1_mean' in agg.columns:
        plots.append(("f1_mean", "Average F1 Score", "f1_vs_model.png"))

    for col, title, fname in plots:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=agg.sort_values(col, ascending=False), x='model_name', y=col, palette='viridis')
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_file = out_dir / fname
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparative analysis plots from knowledge_base.csv")
    parser.add_argument("--kb", default="knowledge_base.csv", help="Path to knowledge_base.csv")
    parser.add_argument("--out", default="outputs", help="Output directory for plots")
    args = parser.parse_args()

    generate_comparative_plots(args.kb, args.out)
