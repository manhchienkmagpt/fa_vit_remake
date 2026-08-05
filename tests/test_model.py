import torch

from favit_m2tr.model import create_favit_m2tr


def test_adapters_start_as_residual_noops_and_model_runs():
    model = create_favit_m2tr(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        train_backbone_norms=False,
        train_cls_token=False,
        m2tr_channels=8,
        m2tr_depth=1,
        m2tr_fusion_hidden_channels=4,
        m2tr_fusion_pool_size=7,
    )
    gam = model.backbone.blocks[0].attn.gam
    tokens = torch.randn(1, 197, model.embed_dim)
    assert torch.count_nonzero(gam(tokens)) == 0
    assert torch.count_nonzero(model.injectors[0].scale) == 0
    assert torch.count_nonzero(model.m2tr_injectors[0].scale) == 0
    with torch.inference_mode():
        logits, features = model(torch.randn(1, 3, 224, 224), return_features=True)
    assert logits.shape == (1, 2)
    assert features.shape == (1, model.embed_dim)


def test_only_adapters_head_and_spatial_branch_are_trainable_when_strictly_frozen():
    model = create_favit_m2tr(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        train_backbone_norms=False,
        train_cls_token=False,
        m2tr_channels=8,
        m2tr_depth=1,
        m2tr_fusion_hidden_channels=4,
        m2tr_fusion_pool_size=7,
    )
    assert not model.backbone.patch_embed.proj.weight.requires_grad
    assert model.backbone.blocks[0].attn.gam.up.weight.requires_grad
    assert model.m2tr_branch.stages[0].frequency.filter.complex_weight.requires_grad
    assert model.m2tr_injectors[0].scale.requires_grad
    assert model.head.weight.requires_grad
