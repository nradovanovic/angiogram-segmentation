"""
Preprocessing and exploratory utilities for angiogram segmentation dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import (
    AngiogramSegmentationDataset,
    create_dataloaders,
    create_datasets,
    create_splits,
    load_image_mask_pairs,
)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across ``random``, ``numpy`` and ``torch``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe_dataset(pairs: Sequence[Tuple[Path, Path]]) -> dict:
    """
    Compute simple dataset statistics.

    Returns a dictionary containing total counts and basic image statistics.
    """

    import imageio.v2 as imageio

    stats = {
        "num_samples": len(pairs),
        "image_mean": 0.0,
        "image_std": 0.0,
        "mask_mean": 0.0,
    }
    if not pairs:
        return stats

    means: List[float] = []
    stds: List[float] = []
    mask_means: List[float] = []
    for image_path, mask_path in pairs:
        image = imageio.imread(image_path).astype(np.float32)
        mask = imageio.imread(mask_path).astype(np.float32)
        means.append(image.mean())
        stds.append(image.std())
        mask_means.append(mask.mean() / 255.0)

    stats["image_mean"] = float(np.mean(means))
    stats["image_std"] = float(np.mean(stds))
    stats["mask_mean"] = float(np.mean(mask_means))
    return stats


def visualize_samples(
    dataset: AngiogramSegmentationDataset,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot ``num_samples`` random images and masks from the dataset.

    Parameters
    ----------
    dataset:
        Dataset to sample from.
    num_samples:
        Number of examples to plot.
    figsize:
        Matplotlib figure size.
    save_path:
        If provided, save the figure to this path instead of (or in addition to)
        showing it interactively.
    show:
        Whether to call ``plt.show()``. Automatically disabled when running in
        non-interactive environments by setting ``show=False``.
    """

    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    fig, axes = plt.subplots(len(indices), 2, figsize=figsize)
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, idx in zip(axes, indices):
        sample = dataset[idx]
        image = sample["image"].squeeze().numpy()
        mask = sample["mask"].squeeze().numpy()
        row[0].imshow(image, cmap="gray")
        row[0].set_title("Image")
        row[0].axis("off")
        row[1].imshow(image, cmap="gray")
        row[1].imshow(mask, cmap="jet", alpha=0.4)
        row[1].set_title("Mask Overlay")
        row[1].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = [
    "seed_everything",
    "describe_dataset",
    "visualize_samples",
    "load_image_mask_pairs",
    "create_splits",
    "create_datasets",
    "create_dataloaders",
]


