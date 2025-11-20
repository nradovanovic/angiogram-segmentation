"""
Losses and metrics for segmentation models.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F


def dice_coefficient(
    preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


def iou_score(
    preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1)
    union = total - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean()


def precision_recall(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    tp = (preds * targets).sum(dim=[1, 2, 3])
    fp = (preds * (1 - targets)).sum(dim=[1, 2, 3])
    fn = ((1 - preds) * targets).sum(dim=[1, 2, 3])

    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return precision.mean(), recall.mean()


def dice_loss(
    preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    preds = torch.sigmoid(preds)
    preds_flat = preds.contiguous().view(preds.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    loss = 1 - ((2 * intersection + smooth) / (union + smooth))
    return loss.mean()


def bce_dice_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(preds, targets)
    d_loss = dice_loss(preds, targets)
    return bce + d_loss


def focal_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.8,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    preds_prob = torch.sigmoid(preds)
    ce_loss = F.binary_cross_entropy(preds_prob, targets, reduction="none")
    pt = torch.where(targets == 1, preds_prob, 1 - preds_prob)
    loss = ce_loss * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
        loss = alpha_factor * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    dice = dice_coefficient(preds, targets)
    iou = iou_score(preds, targets)
    precision, recall = precision_recall(preds, targets)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


