from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from favit.config import build_model_from_config, load_config, resolve_device, seed_everything
from favit.data import FaceTransform, FrameFaceDataset, PairedFaceDataset, paired_collate
from favit.engine import evaluate_video_level, train_one_epoch
from favit.losses import FineGrainedAdaptiveLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FA-ViT on paired FF++ face frames")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:0")
    return parser.parse_args()


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device or config.get("device", "cuda"))
    data_config = config["data"]
    train_config = config["train"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_batch_size = int(train_config["image_batch_size"])
    if image_batch_size % 2:
        raise ValueError("image_batch_size must be even because every item is a fake-real pair")
    train_transform = FaceTransform(
        image_size=int(data_config.get("image_size", 224)),
        horizontal_flip=float(data_config.get("horizontal_flip", 0.0)),
    )
    eval_transform = FaceTransform(image_size=int(data_config.get("image_size", 224)))
    train_dataset = PairedFaceDataset(
        data_config["train_pairs"], data_config["root"], train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=image_batch_size // 2,
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=paired_collate,
    )
    val_loader = None
    if data_config.get("val_frames"):
        val_dataset = FrameFaceDataset(
            data_config["val_frames"], data_config["root"], eval_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=image_batch_size,
            shuffle=False,
            num_workers=int(data_config.get("num_workers", 8)),
            pin_memory=device.type == "cuda",
        )

    model = build_model_from_config(config["model"]).to(device)
    print(json.dumps(model.trainable_parameter_summary(), indent=2))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config.get("lr_step_size", 5)),
        gamma=float(train_config.get("lr_gamma", 0.5)),
    )
    loss_config = config["loss"]
    fal_criterion = FineGrainedAdaptiveLoss(
        scale=float(loss_config.get("fal_scale", 24.0)),
        margin=float(loss_config.get("fal_margin", 0.25)),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 0
    best_auc = float("-inf")
    resume = args.resume or train_config.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_auc", best_auc))

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(train_config["epochs"])):
        warmup_epochs = int(loss_config.get("fal_warmup_epochs", 1))
        fal_weight = (
            0.0
            if epoch < warmup_epochs
            else float(loss_config.get("fal_weight_after_warmup", 1.0))
        )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, fal_criterion, fal_weight, device, scaler
        )
        val_metrics = evaluate_video_level(model, val_loader, device) if val_loader else {}
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "fal_weight": fal_weight,
            "train": train_metrics,
            "validation": val_metrics,
        }
        print(json.dumps(record, indent=2))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        current_auc = float(val_metrics.get("video_auc", -train_metrics["loss"]))
        # Advance the epoch-based scheduler before serializing so resume starts
        # with exactly the learning rate of the next epoch.
        scheduler.step()
        improved = current_auc > best_auc
        best_auc = max(best_auc, current_auc)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_auc": best_auc,
            "config": config,
        }
        if improved:
            save_checkpoint(output_dir / "best.pt", state)
        save_checkpoint(output_dir / "last.pt", state)


if __name__ == "__main__":
    main()
