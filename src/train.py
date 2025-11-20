"""
Training script for coronary angiogram segmentation models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.metrics import bce_dice_loss, compute_metrics, dice_loss, focal_loss
from src.models import TransUNet, UNet, UNet3Plus, UNetPlusPlus
from src.preprocessing import create_dataloaders, create_datasets, seed_everything
from src.utils.env import resolve_data_dir


LOSS_MAP = {
    "dice": dice_loss,
    "bce_dice": bce_dice_loss,
    "focal": focal_loss,
}

MODEL_MAP = {
    "unet": UNet,
    "unetpp": UNetPlusPlus,
    "unet3plus": UNet3Plus,
    "transunet": TransUNet,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation models on angiograms.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Database_134_Angiograms",
        help="Directory containing angiogram images and masks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Resize all images to this size (height width).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["unetpp", "unet3plus", "transunet"],
        choices=list(MODEL_MAP.keys()),
        help="Models to train.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loss", type=str, default="bce_dice", choices=list(LOSS_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)


def instantiate_model(name: str, image_size: Tuple[int, int]) -> nn.Module:
    model_cls = MODEL_MAP[name]
    if name == "transunet":
        return model_cls(img_size=image_size)
    return model_cls()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[amp.GradScaler],
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_metrics = compute_metrics(outputs.detach(), masks)
        running_loss += loss.item()
        for key in metrics_sum:
            metrics_sum[key] += float(batch_metrics[key])
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_metrics = compute_metrics(outputs, masks)
            running_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += float(batch_metrics[key])
            num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def fit_model(
    model_name: str,
    args: argparse.Namespace,
    datasets: Tuple[
        torch.utils.data.Dataset,
        torch.utils.data.Dataset,
        torch.utils.data.Dataset,
    ],
) -> Dict[str, float]:
    train_loader, val_loader, test_loader = create_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device)
    model = instantiate_model(model_name, tuple(args.image_size)).to(device)
    criterion = LOSS_MAP[args.loss]
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val_dice = 0.0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Epoch {epoch}/{args.epochs} - Model: {model_name}")
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log_entry)
        print(json.dumps(log_entry, indent=2))

        current_val_dice = val_metrics["dice"]
        if current_val_dice > best_val_dice + 1e-4:
            best_val_dice = current_val_dice
            epochs_no_improve = 0
            checkpoint_path = (
                Path(args.output_dir)
                / "checkpoints"
                / f"{model_name}_best.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # Load best checkpoint for testing
    best_path = Path(args.output_dir) / "checkpoints" / f"{model_name}_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint from {best_path}")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print("Test metrics:", test_metrics)

    metrics_path = (
        Path(args.output_dir) / "metrics" / f"{model_name}_metrics.json"
    )
    with metrics_path.open("w") as f:
        json.dump(
            {
                "history": history,
                "best_val_dice": best_val_dice,
                "test_loss": test_loss,
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            },
            f,
            indent=2,
        )
    return {k: float(v) for k, v in test_metrics.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir)

    data_dir = resolve_data_dir(args.data_dir)

    datasets = create_datasets(
        data_dir,
        image_size=tuple(args.image_size),
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
        augment=True,
    )

    summary: Dict[str, Dict[str, float]] = {}
    for model_name in args.models:
        metrics = fit_model(model_name, args, datasets)
        summary[model_name] = metrics

    summary_path = output_dir / "metrics" / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary metrics to", summary_path)


if __name__ == "__main__":
    main()


