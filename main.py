from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from build_batches import build_plantseg_batches
from rmi_torch_model import RMIResNetModel
from server_runtime import (
    array_from_tensor,
    collate_batch,
    NpzBatchDataset,
    RunningMaskMetrics,
    copy_config_snapshot,
    ensure_output_dirs,
    load_config,
    load_vocab,
    repo_root,
    resize_prediction_to_mask,
    save_metrics,
    save_prediction_mask,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the PyTorch RMI plantseg pipeline.")
    parser.add_argument("--config", default="configs/plantseg_rmi_resnet.yaml")
    parser.add_argument("--mode", choices=["build", "train", "test"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for test mode.")
    return parser.parse_args()


def pick_device(config: Dict) -> torch.device:
    requested = config["training"]["device"]
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(config: Dict, split: str, shuffle: bool) -> DataLoader:
    dataset = NpzBatchDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_batch,
    )


def ensure_batches(config: Dict) -> None:
    batch_root = Path(config["dataset"]["batch_dir"])
    if not all((batch_root / f"{split}_batch").exists() for split in ("train", "val", "test")):
        build_plantseg_batches(config)


def create_model(config: Dict) -> RMIResNetModel:
    vocab = load_vocab(config["dataset"]["vocab_path"])
    return RMIResNetModel(config, vocab_size=len(vocab))


def save_checkpoint(config: Dict, model: torch.nn.Module, optimizer: Adam, epoch: int, best_miou: float, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_miou": best_miou,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, optimizer: Optional[Adam] = None) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def evaluate(config: Dict, model: torch.nn.Module, device: torch.device, split: str, export_masks: bool) -> Dict[str, float]:
    loader = make_loader(config, split=split, shuffle=False)
    metrics = RunningMaskMetrics()
    model.eval()

    mask_dir = config["outputs"]["test_mask_dir"] if export_masks else None
    threshold = float(config["evaluation"]["threshold"])

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split} eval", leave=False):
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            logits = model(image, text).squeeze(1)
            pred_raw = array_from_tensor((torch.sigmoid(logits) >= threshold).to(torch.float32), np.float32)

            for idx in range(pred_raw.shape[0]):
                gt_mask = array_from_tensor(batch["gt_mask"][idx], np.float32)
                pred_mask = resize_prediction_to_mask(pred_raw[idx], gt_mask.shape)
                metrics.update(pred_mask, gt_mask)
                if mask_dir:
                    save_prediction_mask(pred_mask, mask_dir, batch["mask_relpath"][idx])

    result = metrics.compute()
    result["split"] = split
    return result


def train(config: Dict, config_path: str) -> None:
    ensure_output_dirs(config)
    ensure_batches(config)
    set_seed(int(config["training"]["seed"]))
    copy_config_snapshot(config, config_path)

    device = pick_device(config)
    model = create_model(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    loader = make_loader(config, split="train", shuffle=True)
    total_steps = max(len(loader) * int(config["training"]["epochs"]), 1)
    base_lr = float(config["training"]["learning_rate"])
    min_lr = float(config["training"]["min_learning_rate"])

    def poly_lambda(step: int) -> float:
        progress = min(step / total_steps, 1.0)
        current_lr = ((base_lr - min_lr) * ((1.0 - progress) ** 0.9)) + min_lr
        return current_lr / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)

    best_miou = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            target = batch["train_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image, text)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(len(loader), 1)
        val_metrics = evaluate(config, model, device, split="val", export_masks=False)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = avg_loss
        save_metrics(val_metrics, Path(config["outputs"]["metrics_dir"]) / f"val_epoch_{epoch:03d}.json")

        save_checkpoint(config, model, optimizer, epoch, best_miou, config["outputs"]["checkpoint_last"])
        if val_metrics["mIoU"] >= best_miou:
            best_miou = float(val_metrics["mIoU"])
            best_metrics = val_metrics
            save_checkpoint(config, model, optimizer, epoch, best_miou, config["outputs"]["checkpoint_best"])

    if best_metrics:
        save_metrics(best_metrics, Path(config["outputs"]["metrics_dir"]) / "best_val_metrics.json")


def test(config: Dict, checkpoint_path: Optional[str]) -> None:
    ensure_output_dirs(config)
    ensure_batches(config)
    device = pick_device(config)
    model = create_model(config).to(device)

    resolved_checkpoint = checkpoint_path or config["outputs"]["checkpoint_best"]
    load_checkpoint(model, resolved_checkpoint)
    test_metrics = evaluate(
        config,
        model,
        device,
        split="test",
        export_masks=bool(config["evaluation"]["export_test_masks"]),
    )
    test_metrics["checkpoint"] = str(Path(resolved_checkpoint).resolve())
    save_metrics(test_metrics, Path(config["outputs"]["metrics_dir"]) / "test_metrics.json")
    print(json.dumps(test_metrics, indent=2))


def main() -> None:
    args = parse_args()
    config_path = str((repo_root() / args.config).resolve()) if not Path(args.config).is_absolute() else args.config
    config = load_config(config_path)

    if args.mode == "build":
        build_plantseg_batches(config)
    elif args.mode == "train":
        train(config, config_path)
    elif args.mode == "test":
        test(config, args.checkpoint)


if __name__ == "__main__":
    main()
