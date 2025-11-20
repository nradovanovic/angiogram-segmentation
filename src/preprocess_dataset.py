"""
Command-line preprocessing utility to create persistent dataset splits
and optionally export resized numpy arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np

from src.data.dataset import (
    AngiogramSegmentationDataset,
    create_splits,
    default_resize_transform,
    load_image_mask_pairs,
)
from src.preprocessing import seed_everything
from src.utils.env import resolve_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits and optionally export preprocessed arrays."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("Database_134_Angiograms"),
        help="Directory containing angiogram *.pgm files and *_gt.pgm masks.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/preprocessed"),
        help="Directory where splits, manifests, and exported arrays will be stored.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("HEIGHT", "WIDTH"),
        help="Target height and width for resizing images/masks.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling the split.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="If set, export resized numpy arrays for each split.",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "npz"),
        default="npy",
        help="File format used when exporting arrays.",
    )
    return parser.parse_args()


def normalize(image: np.ndarray) -> np.ndarray:
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def export_split(
    split_name: str,
    pairs: Iterable[Tuple[Path, Path]],
    resize_transform,
    output_dir: Path,
    file_format: str,
) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    images_dir = output_dir / split_name / "images"
    masks_dir = output_dir / split_name / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        image = imageio.imread(image_path).astype(np.float32)
        mask = imageio.imread(mask_path).astype(np.float32)

        image = normalize(image)
        mask = (mask > 0).astype(np.float32)

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        if resize_transform is not None:
            transformed = resize_transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = np.transpose(image, (2, 0, 1))  # to CHW
        mask = np.transpose(mask, (2, 0, 1))

        stem = image_path.stem.replace("_gt", "")
        image_file = images_dir / f"{stem}.{file_format}"
        mask_file = masks_dir / f"{stem}.{file_format}"

        if file_format == "npy":
            np.save(image_file, image, allow_pickle=False)
            np.save(mask_file, mask, allow_pickle=False)
        else:
            np.savez_compressed(image_file, image=image)
            np.savez_compressed(mask_file, mask=mask)

        manifest.append(
            {
                "stem": stem,
                "original_image": str(image_path),
                "original_mask": str(mask_path),
                "image_file": str(image_file),
                "mask_file": str(mask_file),
            }
        )
    return manifest


def save_manifest(manifest: Dict[str, List[Dict[str, str]]], output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def save_splits(splits: Dict[str, List[Tuple[Path, Path]]], output_dir: Path) -> Path:
    serializable = {
        split: [
            {"image": str(image_path), "mask": str(mask_path)}
            for image_path, mask_path in pairs
        ]
        for split, pairs in splits.items()
    }
    splits_path = output_dir / "splits.json"
    with splits_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    return splits_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Collecting image/mask pairs from {data_dir}...")
    pairs = load_image_mask_pairs(data_dir)
    splits = create_splits(
        pairs,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    splits_dict = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    splits_path = save_splits(splits_dict, output_dir)
    print(f"Saved split definitions to {splits_path}")

    if not args.export:
        print("Export flag not set; skipping array export.")
        return

    resize_transform = default_resize_transform(tuple(args.image_size))
    manifest: Dict[str, List[Dict[str, str]]] = {}
    for split_name, split_pairs in splits_dict.items():
        print(f"Exporting {split_name} split with {len(split_pairs)} samples...")
        manifest[split_name] = export_split(
            split_name,
            split_pairs,
            resize_transform,
            output_dir=output_dir,
            file_format=args.format,
        )

    manifest_path = save_manifest(manifest, output_dir)
    print(f"Saved export manifest to {manifest_path}")


if __name__ == "__main__":
    main()


