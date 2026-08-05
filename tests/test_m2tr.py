import pytest
import torch

from favit_m2tr.m2tr import (
    CrossModalFusion,
    LearnableFrequencyFilter,
    M2TRFeatureBranch,
    MultiScalePatchAttention,
)


def test_multi_scale_attention_preserves_spatial_shape():
    module = MultiScalePatchAttention(8, patch_sizes=(8, 4, 2, 1))
    output = module(torch.randn(2, 8, 8, 8))
    assert output.shape == (2, 8, 8, 8)


def test_multi_scale_attention_rejects_incompatible_grid():
    module = MultiScalePatchAttention(8, patch_sizes=(8, 4, 2, 1))
    with pytest.raises(ValueError, match="not divisible"):
        module(torch.randn(1, 8, 10, 10))


def test_frequency_filter_is_differentiable_and_preserves_dtype_and_shape():
    module = LearnableFrequencyFilter(channels=4, height=8, width=8)
    inputs = torch.randn(2, 4, 8, 8, requires_grad=True)
    output = module(inputs)
    output.square().mean().backward()
    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    assert module.complex_weight.grad is not None


def test_cross_modal_fusion_uses_pooled_context_without_changing_grid():
    module = CrossModalFusion(channels=8, hidden_channels=4, pool_size=4)
    rgb = torch.randn(2, 8, 8, 8)
    frequency = torch.randn_like(rgb)
    assert module(rgb, frequency).shape == rgb.shape


def test_feature_branch_emits_vit_aligned_tokens():
    branch = M2TRFeatureBranch(
        input_channels=8,
        embed_dim=24,
        image_size=32,
        channels=8,
        depth=1,
        patch_sizes=(8, 4, 2, 1),
        fusion_hidden_channels=4,
        fusion_pool_size=4,
    )
    output = branch(torch.randn(2, 8, 8, 8), token_grid=2)
    assert output.shape == (2, 4, 24)

