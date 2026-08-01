import torch

from favit.losses import FineGrainedAdaptiveLoss


def test_fal_rewards_aligned_real_and_separated_fake():
    criterion = FineGrainedAdaptiveLoss(scale=24, margin=0.25)
    prototype = torch.tensor([1.0, 0.0])
    good = criterion(
        prototype,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
    )
    bad = criterion(
        prototype,
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    assert good < bad


def test_fal_accepts_batched_prototype():
    criterion = FineGrainedAdaptiveLoss()
    real = torch.randn(3, 8)
    fake = torch.randn(3, 8)
    prototype = torch.randn(3, 8)
    assert criterion(prototype, real, fake).ndim == 0

