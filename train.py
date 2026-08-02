from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from favit.config import build_model_from_config, load_config, resolve_device, seed_everything
from favit.data import FaceTransform, FrameFaceDataset, PairedFaceDataset, paired_collate
from favit.engine import evaluate_video_level, train_one_epoch
from favit.losses import FineGrainedAdaptiveLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FA-ViT on paired FF++ face frames")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint (normally outputs/.../last.pt)",
    )
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:0")
    return parser.parse_args()


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_random_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict | None) -> None:
    """Restore RNG state when available while accepting older checkpoints."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_frame_loader(
    manifest: str | Path,
    data_config: dict,
    transform: FaceTransform,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset = FrameFaceDataset(manifest, data_config["root"], transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )


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
    if not data_config.get("celebdf_test_frames"):
        raise ValueError("data.celebdf_test_frames is required for evaluation during training")
    celebdf_test_loader = make_frame_loader(
        data_config["celebdf_test_frames"],
        data_config,
        eval_transform,
        image_batch_size,
        device,
    )

    resume_value = args.resume or train_config.get("resume")
    resume_path = Path(resume_value) if resume_value else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    # A resume checkpoint already contains the complete model, so avoid loading
    # or downloading the pretrained backbone before replacing all its weights.
    model = build_model_from_config(
        config["model"], pretrained=False if resume_path is not None else None
    ).to(device)
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
    best_celebdf_auc = float("-inf")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_celebdf_auc = float(checkpoint.get("best_celebdf_auc", best_celebdf_auc))
        restore_random_state(checkpoint.get("random_state"))
        print(
            "resume_checkpoint: "
            f"path={resume_path} next_epoch={start_epoch + 1} "
            f"best_celebdf_auc={best_celebdf_auc:.6f}"
        )
        if "best_celebdf_auc" not in checkpoint:
            print(
                "resume_checkpoint: best_celebdf_auc is unavailable in this older "
                "checkpoint; the best CelebDF AUC is reset"
            )

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
        celebdf_test_metrics = evaluate_video_level(
            model, celebdf_test_loader, device, description="test CelebDF"
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "fal_weight": fal_weight,
            "train": train_metrics,
            "celebdf_test": celebdf_test_metrics,
        }
        print(json.dumps(record, indent=2))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        current_celebdf_auc = float(celebdf_test_metrics["video_auc"])
        # Advance the epoch-based scheduler before serializing so resume starts
        # with exactly the learning rate of the next epoch.
        scheduler.step()
        improved = current_celebdf_auc > best_celebdf_auc
        best_celebdf_auc = max(best_celebdf_auc, current_celebdf_auc)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            # Keep best_auc as a generic alias for external checkpoint consumers.
            "best_auc": best_celebdf_auc,
            "best_celebdf_auc": best_celebdf_auc,
            "celebdf_test_metrics": celebdf_test_metrics,
            "random_state": capture_random_state(),
            "config": config,
        }
        if improved:
            best_path = output_dir / "best.pt"
            save_checkpoint(best_path, state)
            print(
                "save_best_checkpoint: "
                f"path={best_path} epoch={epoch + 1} "
                f"celebdf_test_auc={current_celebdf_auc:.6f}"
            )
        save_checkpoint(output_dir / "last.pt", state)


if __name__ == "__main__":
    main()
