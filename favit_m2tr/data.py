from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class FaceTransform:
    """224x224 RGB, optional shared flip, and the public-code [-1, 1] normalization."""

    def __init__(self, image_size: int = 224, horizontal_flip: float = 0.0) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip

    def sample_flip(self) -> bool:
        return random.random() < self.horizontal_flip

    def __call__(self, image: Image.Image, flip: bool = False) -> Tensor:
        image = image.convert("RGB")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if flip:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _resolve(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


class PairedFaceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fine-grained FF++ fake/target-real frame pairs for CE + FAL training."""

    REQUIRED_COLUMNS = {"fake_path", "real_path"}

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        transform: FaceTransform,
    ) -> None:
        self.rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - self.rows[0].keys()
        if missing:
            raise ValueError(f"paired manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.rows[index]
        with Image.open(_resolve(self.data_root, row["fake_path"])) as image:
            fake_image = image.copy()
        with Image.open(_resolve(self.data_root, row["real_path"])) as image:
            real_image = image.copy()
        # The same geometric transform preserves fine-grained correspondence.
        flip = self.transform.sample_flip()
        return self.transform(fake_image, flip), self.transform(real_image, flip)


class FrameFaceDataset(Dataset[tuple[Tensor, int, str]]):
    """Independent frames with a stable video id for video-level evaluation."""

    REQUIRED_COLUMNS = {"path", "label", "video_id"}

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        transform: FaceTransform,
    ) -> None:
        self.rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - self.rows[0].keys()
        if missing:
            raise ValueError(f"frame manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        row = self.rows[index]
        with Image.open(_resolve(self.data_root, row["path"])) as image:
            tensor = self.transform(image.copy())
        return tensor, int(row["label"]), row["video_id"]


def paired_collate(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return [fake..., real...] images and explicit feature pairing indices."""

    fake = torch.stack([item[0] for item in batch])
    real = torch.stack([item[1] for item in batch])
    images = torch.cat((fake, real), dim=0)
    labels = torch.cat(
        (
            torch.ones(len(batch), dtype=torch.long),
            torch.zeros(len(batch), dtype=torch.long),
        )
    )
    pair_count = torch.tensor(len(batch), dtype=torch.long)
    return images, labels, pair_count

