from __future__ import annotations

import math
from collections.abc import Sequence

import timm
import torch
from torch import Tensor, nn


class SeparableConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, padding: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class GlobalAdaptiveModule(nn.Module):
    """GAM: bottleneck 1x1, spatial 3x3, then zero-initialized Q/K/V 1x1."""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # Linear layers are equivalent to 1x1 convolutions on the token grid.
        self.down = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.spatial = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.up = nn.Linear(hidden_dim, embed_dim * 3, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        patch_tokens = tokens - 1
        grid = math.isqrt(patch_tokens)
        if grid * grid != patch_tokens:
            raise ValueError(f"GAM requires a square patch grid, got {patch_tokens} patches")

        reduced = self.down(x)
        patch = reduced[:, 1:].reshape(batch, grid, grid, self.hidden_dim)
        patch = self.spatial(patch.permute(0, 3, 1, 2))
        patch = patch.permute(0, 2, 3, 1).reshape(batch, patch_tokens, self.hidden_dim)

        # This mirrors the public implementation: the class token is treated as
        # a padded 1x1 feature map by the same convolution.
        cls = reduced[:, :1].reshape(batch, 1, 1, self.hidden_dim).permute(0, 3, 1, 2)
        cls = self.spatial(cls).permute(0, 2, 3, 1).reshape(batch, 1, self.hidden_dim)
        delta = self.up(torch.cat((cls, patch), dim=1))
        return delta.reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)


class ForgeryAwareAttention(nn.Module):
    """A timm ViT attention layer augmented with GAM."""

    def __init__(self, attention: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.num_heads = attention.num_heads
        self.scale = attention.scale
        self.qkv = attention.qkv
        self.q_norm = getattr(attention, "q_norm", nn.Identity())
        self.k_norm = getattr(attention, "k_norm", nn.Identity())
        self.attn_drop = attention.attn_drop
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop
        embed_dim = self.qkv.in_features
        self.gam = GlobalAdaptiveModule(embed_dim, self.num_heads, hidden_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if attn_mask is not None:
            raise NotImplementedError("FA-ViT reproduction does not use an attention mask")
        if is_causal:
            raise NotImplementedError("FA-ViT reproduction does not use causal attention")
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dq, dk, dv = self.gam(x).unbind(0)
        q, k, v = self.q_norm(q + dq), self.k_norm(k + dk), v + dv
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class SpatialCNN(nn.Module):
    """One three-convolution spatial block from Table II/public code."""

    def __init__(
        self, in_channels: int, embed_dim: int, projection_kernel: int, stride: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels * 2,
                in_channels * 4,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
        )
        self.project = nn.Conv2d(
            in_channels * 2,
            embed_dim,
            kernel_size=projection_kernel,
            stride=projection_kernel,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.block(x)
        batch = x.shape[0]
        tokens = self.project(x).flatten(2).transpose(1, 2)
        return x, tokens


class LocalAdaptiveAttention(nn.Module):
    """LAM attention combining learned cross-attention and quadratic locality."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by number of heads")
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.position_projection = nn.Linear(3, num_heads, bias=False)
        self.gating = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer("relative_indices", torch.empty(0), persistent=False)
        self._initialize_local_directions()

    def _initialize_local_directions(self) -> None:
        nn.init.trunc_normal_(self.position_projection.weight, std=0.02)
        directions = [
            (vertical, horizontal)
            for vertical in (-1, 0, 1)
            for horizontal in (-1, 0, 1)
        ]
        with torch.no_grad():
            for head in range(self.num_heads):
                psi1, psi2 = directions[head % len(directions)]
                # Feature order is (delta_x, delta_y, squared_distance).
                self.position_projection.weight[head] = torch.tensor(
                    [2.0 * psi2, 2.0 * psi1, -1.0]
                )

    def _get_relative_indices(self, patches: int, device: torch.device) -> Tensor:
        if self.relative_indices.numel() and self.relative_indices.shape[1] == patches:
            return self.relative_indices.to(device)
        grid = math.isqrt(patches)
        if grid * grid != patches:
            raise ValueError(f"LAM requires a square spatial grid, got {patches} tokens")
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        delta = coordinates[None, :, :] - coordinates[:, None, :]
        distance = delta.square().sum(dim=-1, keepdim=True)
        relative = torch.cat((delta[..., 1:], delta[..., :1], distance), dim=-1).float()
        self.relative_indices = relative.unsqueeze(0).to(device)
        return self.relative_indices

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        batch, query_tokens, channels = query.shape
        patches = context.shape[1]
        head_dim = channels // self.num_heads
        q = self.q(query).reshape(batch, query_tokens, self.num_heads, head_dim).transpose(1, 2)
        k = self.k(context).reshape(batch, patches, self.num_heads, head_dim).transpose(1, 2)
        v = self.v(context).reshape(batch, patches, self.num_heads, head_dim).transpose(1, 2)

        global_attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        relative = self._get_relative_indices(patches, query.device)
        local_logits = self.position_projection(relative).permute(0, 3, 1, 2)
        local_attention = local_logits.softmax(dim=-1)

        # The class token remains global. Patch queries use equation (5), with
        # sigma initialized to zero so the initial mixture is exactly 0.5/0.5.
        cls_attention = global_attention[:, :, :1]
        patch_global = global_attention[:, :, 1:]
        gate = torch.sigmoid(self.gating).view(1, -1, 1, 1)
        patch_attention = (1.0 - gate) * patch_global + gate * local_attention
        attention = torch.cat((cls_attention, patch_attention), dim=2)
        output = (attention @ v).transpose(1, 2).reshape(batch, query_tokens, channels)
        return self.proj(output)


class LocalInjector(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        norm = lambda: nn.LayerNorm(dim, eps=1e-6)
        self.query_norm = norm()
        self.context_norm = norm()
        self.output_norm = norm()
        self.attention = LocalAdaptiveAttention(dim, num_heads)
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        adapted = self.attention(self.query_norm(query), self.context_norm(context))
        return query + self.scale * self.output_norm(adapted)


class ForgeryAwareViT(nn.Module):
    """FA-ViT using a timm ViT backbone and paper/public-code adaptive modules."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        gam_reduction: int = 2,
        inject_layers: Sequence[int] = (0, 3, 6),
        train_backbone_norms: bool = True,
        train_cls_token: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "blocks") or not hasattr(backbone, "patch_embed"):
            raise TypeError("backbone must be a timm VisionTransformer")
        self.backbone = backbone
        self.embed_dim = int(backbone.embed_dim)
        self.inject_layers = tuple(int(index) for index in inject_layers)
        num_heads = int(backbone.blocks[0].attn.num_heads)
        hidden_dim = self.embed_dim // gam_reduction

        for block in self.backbone.blocks:
            block.attn = ForgeryAwareAttention(block.attn, hidden_dim)

        self.head = nn.Linear(self.embed_dim, num_classes, bias=False)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SeparableConv2d(32),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # For a 224x224 input these yield three 14x14 token maps.
        self.spatial_blocks = nn.ModuleList(
            (
                SpatialCNN(32, self.embed_dim, projection_kernel=4, stride=1),
                SpatialCNN(64, self.embed_dim, projection_kernel=2, stride=2),
                SpatialCNN(128, self.embed_dim, projection_kernel=1, stride=2),
            )
        )
        if len(self.inject_layers) != len(self.spatial_blocks):
            raise ValueError("the paper architecture requires exactly three injection layers")
        self.injectors = nn.ModuleList(
            LocalInjector(self.embed_dim, num_heads) for _ in self.inject_layers
        )
        self._set_trainable_parameters(train_backbone_norms, train_cls_token)

    def _set_trainable_parameters(
        self, train_backbone_norms: bool, train_cls_token: bool
    ) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in self.backbone.blocks:
            for parameter in block.attn.gam.parameters():
                parameter.requires_grad = True
        if train_backbone_norms:
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = train_cls_token
        for module in (self.head, self.spatial_stem, self.spatial_blocks, self.injectors):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _embed(self, images: Tensor) -> Tensor:
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        return x

    def forward_features(self, images: Tensor) -> Tensor:
        spatial = self.spatial_stem(images)
        tokens = self._embed(images)
        injection_index = 0
        for block_index, block in enumerate(self.backbone.blocks):
            if block_index in self.inject_layers:
                spatial, spatial_tokens = self.spatial_blocks[injection_index](spatial)
                if spatial_tokens.shape[1] != tokens.shape[1] - 1:
                    raise ValueError(
                        "spatial and ViT token grids differ; FA-ViT expects 224x224 inputs"
                    )
                tokens = self.injectors[injection_index](tokens, spatial_tokens)
                injection_index += 1
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        return tokens[:, 0]

    def forward(
        self, images: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.forward_features(images)
        logits = self.head(features)
        return (logits, features) if return_features else logits

    def trainable_parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_favit(
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    pretrained: bool = True,
    num_classes: int = 2,
    gam_reduction: int = 2,
    inject_layers: Sequence[int] = (0, 3, 6),
    train_backbone_norms: bool = True,
    train_cls_token: bool = True,
) -> ForgeryAwareViT:
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return ForgeryAwareViT(
        backbone=backbone,
        num_classes=num_classes,
        gam_reduction=gam_reduction,
        inject_layers=inject_layers,
        train_backbone_norms=train_backbone_norms,
        train_cls_token=train_cls_token,
    )
