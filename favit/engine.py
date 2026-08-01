from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from .losses import FineGrainedAdaptiveLoss
from .metrics import video_level_metrics


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    fal_criterion: FineGrainedAdaptiveLoss,
    fal_weight: float,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "fal": 0.0, "accuracy": 0.0}
    batches = 0
    use_amp = scaler is not None and scaler.is_enabled()

    for images, labels, pair_count_tensor in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pair_count = int(pair_count_tensor.item())
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, features = model(images, return_features=True)
            ce_loss = F.cross_entropy(logits, labels)
            fake_features = features[:pair_count]
            real_features = features[pair_count:]
            prototype = model.head.weight[0]
            fal_loss = fal_criterion(prototype, real_features, fake_features)
            loss = ce_loss + fal_weight * fal_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["ce"] += float(ce_loss.detach())
        totals["fal"] += float(fal_loss.detach())
        totals["accuracy"] += float((logits.argmax(dim=1) == labels).float().mean())
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


@torch.inference_mode()
def evaluate_video_level(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor, list[str]]],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    video_ids: list[str] = []
    for images, batch_labels, batch_video_ids in tqdm(loader, desc="evaluate", leave=False):
        logits = model(images.to(device, non_blocking=True))
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        labels.extend(batch_labels.tolist())
        video_ids.extend(batch_video_ids)
    return video_level_metrics(probabilities, labels, video_ids)

