from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from favit_m2tr.config import build_model_from_config, load_config, resolve_device
from favit_m2tr.data import FaceTransform, FrameFaceDataset
from favit_m2tr.engine import evaluate_video_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-level FA-ViT/M2TR evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Evaluation must not download ImageNet-21K weights; the checkpoint is complete.
    model_config = checkpoint.get("config", config)["model"]
    model = build_model_from_config(model_config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    data_config = config["data"]
    manifest = args.manifest or data_config["celebdf_test_frames"]
    dataset = FrameFaceDataset(
        manifest,
        data_config["root"],
        FaceTransform(image_size=int(data_config.get("image_size", 224))),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["image_batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_video_level(model, loader, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
