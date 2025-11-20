"""
Utility to aggregate and plot model comparison metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for metrics_file in metrics_dir.glob("*_metrics.json"):
        with metrics_file.open() as f:
            data = json.load(f)
        model_name = metrics_file.stem.replace("_metrics", "")
        test_metrics = data.get("test_metrics", {})
        rows.append(
            {
                "model": model_name,
                **test_metrics,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metric files found in {metrics_dir}")
    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["dice", "iou", "precision", "recall"]:
        if metric not in df.columns:
            continue
        ax = df.plot(
            x="model",
            y=metric,
            kind="bar",
            legend=False,
            rot=0,
            title=f"{metric.upper()} comparison",
        )
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        figure_path = output_dir / f"{metric}_comparison.png"
        plt.savefig(figure_path)
        plt.close()
        print(f"Saved {metric} plot to {figure_path}")


def save_table(df: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics table to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot segmentation metric comparisons.")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="results/metrics",
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to store generated figures and tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    df = load_metrics(metrics_dir)
    output_dir = Path(args.output_dir)
    save_table(df, output_dir)
    plot_metrics(df, output_dir)


if __name__ == "__main__":
    main()


