import torch

from favit.model import create_favit


def test_adapters_start_as_residual_noops_and_model_runs():
    model = create_favit(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        train_backbone_norms=False,
        train_cls_token=False,
    )
    gam = model.backbone.blocks[0].attn.gam
    tokens = torch.randn(1, 197, model.embed_dim)
    assert torch.count_nonzero(gam(tokens)) == 0
    assert torch.count_nonzero(model.injectors[0].scale) == 0
    with torch.inference_mode():
        logits, features = model(torch.randn(1, 3, 224, 224), return_features=True)
    assert logits.shape == (1, 2)
    assert features.shape == (1, model.embed_dim)


def test_only_adapters_head_and_spatial_branch_are_trainable_when_strictly_frozen():
    model = create_favit(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        train_backbone_norms=False,
        train_cls_token=False,
    )
    assert not model.backbone.patch_embed.proj.weight.requires_grad
    assert model.backbone.blocks[0].attn.gam.up.weight.requires_grad
    assert model.head.weight.requires_grad

