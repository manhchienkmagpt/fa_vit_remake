import csv

import torch
from PIL import Image

from favit_m2tr.data import FaceTransform, PairedFaceDataset, paired_collate


def test_paired_dataset_and_collate(tmp_path):
    Image.new("RGB", (24, 24), "red").save(tmp_path / "fake.jpg")
    Image.new("RGB", (24, 24), "blue").save(tmp_path / "real.jpg")
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fake_path", "real_path"])
        writer.writeheader()
        writer.writerow({"fake_path": "fake.jpg", "real_path": "real.jpg"})
    dataset = PairedFaceDataset(manifest, tmp_path, FaceTransform(224))
    images, labels, pair_count = paired_collate([dataset[0]])
    assert images.shape == (2, 3, 224, 224)
    assert torch.equal(labels, torch.tensor([1, 0]))
    assert pair_count.item() == 1
