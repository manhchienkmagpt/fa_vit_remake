from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_model_from_config(model_config: dict[str, Any], pretrained: bool | None = None):
    from .model import create_favit

    return create_favit(
        model_name=model_config["backbone"],
        pretrained=model_config.get("pretrained", True) if pretrained is None else pretrained,
        num_classes=model_config.get("num_classes", 2),
        gam_reduction=model_config.get("gam_reduction", 2),
        inject_layers=model_config.get("inject_layers", [0, 3, 6]),
        train_backbone_norms=model_config.get("train_backbone_norms", True),
        train_cls_token=model_config.get("train_cls_token", True),
    )

