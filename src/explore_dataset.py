"""
Command-line utility for dataset exploration and visualisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.preprocessing import (
    create_datasets,
    describe_dataset,
    load_image_mask_pairs,
    seed_everything,
    visualize_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore coronary angiogram dataset samples and statistics."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("Database_134_Angiograms"),
        help="Directory containing angiogram *.pgm images and *_gt.pgm masks.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/eda"),
        help="Directory where statistics and figures will be stored.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("HEIGHT", "WIDTH"),
        help="Size to which images are resized before visualisation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of samples to visualise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Skip interactive display (useful in headless environments).",
    )
    return parser.parse_args()


def save_stats(stats: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    return stats_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"Loading dataset from {args.data_dir}")
    pairs = load_image_mask_pairs(args.data_dir)
    stats = describe_dataset(pairs)
    stats["image_size"] = list(args.image_size)
    stats["num_samples_visualised"] = args.num_samples

    stats_path = save_stats(stats, args.output_dir)
    print(f"Saved dataset statistics to {stats_path}")

    train_ds, _, _ = create_datasets(
        data_dir=args.data_dir,
        image_size=tuple(args.image_size),
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
        augment=False,
    )

    figure_path = args.output_dir / "sample_overlays.png"
    visualize_samples(
        train_ds,
        num_samples=args.num_samples,
        figsize=(12, max(4, args.num_samples * 2)),
        save_path=figure_path,
        show=not args.no_show,
    )
    print(f"Saved sample visualisations to {figure_path}")


if __name__ == "__main__":
    main()


