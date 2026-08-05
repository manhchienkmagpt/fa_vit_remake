from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeedForward2d(nn.Module):
    """A spatial feed-forward block used after M2TR attention/filtering."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _patch_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    scale = query.shape[-1] ** -0.5
    weights = ((query * scale) @ key.transpose(-2, -1)).softmax(dim=-1)
    return weights @ value


class MultiScalePatchAttention(nn.Module):
    """M2TR patch-wise attention with one spatial scale per channel head.

    Unlike conventional multi-head attention, a head token is an entire spatial
    patch. Large heads compare coarse face regions while small heads compare
    localized texture regions.
    """

    def __init__(self, channels: int, patch_sizes: Sequence[int]) -> None:
        super().__init__()
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        if not self.patch_sizes or any(size <= 0 for size in self.patch_sizes):
            raise ValueError("patch_sizes must contain positive integers")
        if channels % len(self.patch_sizes):
            raise ValueError("channels must be divisible by the number of patch scales")
        self.head_channels = channels // len(self.patch_sizes)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    @staticmethod
    def _to_patches(x: Tensor, patch_size: int) -> tuple[Tensor, tuple[int, int]]:
        batch, channels, height, width = x.shape
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"feature map {height}x{width} is not divisible by patch size {patch_size}"
            )
        rows, columns = height // patch_size, width // patch_size
        patches = x.reshape(
            batch, channels, rows, patch_size, columns, patch_size
        ).permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, rows * columns, -1), (rows, columns)

    @staticmethod
    def _from_patches(
        patches: Tensor,
        grid: tuple[int, int],
        channels: int,
        patch_size: int,
    ) -> Tensor:
        batch = patches.shape[0]
        rows, columns = grid
        return patches.reshape(
            batch, rows, columns, channels, patch_size, patch_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, rows * patch_size, columns * patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        query_heads = self.query(x).chunk(len(self.patch_sizes), dim=1)
        key_heads = self.key(x).chunk(len(self.patch_sizes), dim=1)
        value_heads = self.value(x).chunk(len(self.patch_sizes), dim=1)
        outputs: list[Tensor] = []
        for patch_size, query, key, value in zip(
            self.patch_sizes, query_heads, key_heads, value_heads, strict=True
        ):
            query_patches, grid = self._to_patches(query, patch_size)
            key_patches, _ = self._to_patches(key, patch_size)
            value_patches, _ = self._to_patches(value, patch_size)
            attended = _patch_attention(query_patches, key_patches, value_patches)
            outputs.append(
                self._from_patches(attended, grid, self.head_channels, patch_size)
            )
        return self.output(torch.cat(outputs, dim=1))


class MultiScaleTransformerBlock(nn.Module):
    """Residual multi-scale transformer from Section 3.1 of M2TR."""

    def __init__(
        self, channels: int, patch_sizes: Sequence[int], dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attention = MultiScalePatchAttention(channels, patch_sizes)
        self.feed_forward = FeedForward2d(channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class LearnableFrequencyFilter(nn.Module):
    """Learnable complex global filter in the real-FFT domain.

    FFT math is intentionally kept in FP32 under AMP. The filter is resized in
    frequency space when needed, allowing evaluation at another compatible
    resolution without creating new parameters.
    """

    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        frequency_width = width // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(channels, height, frequency_width, 2, dtype=torch.float32)
            * 0.02
        )

    def _weight_for(self, height: int, width: int) -> Tensor:
        weight = self.complex_weight
        if weight.shape[1:3] != (height, width):
            components = weight.permute(3, 0, 1, 2)
            components = F.interpolate(
                components, size=(height, width), mode="bilinear", align_corners=False
            )
            weight = components.permute(1, 2, 3, 0)
        return torch.view_as_complex(weight.contiguous())

    def forward(self, x: Tensor) -> Tensor:
        output_dtype = x.dtype
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        weight = self._weight_for(height, spectrum.shape[-1])
        filtered = torch.fft.irfft2(
            spectrum * weight.unsqueeze(0),
            s=(height, width),
            dim=(-2, -1),
            norm="ortho",
        )
        return filtered.to(output_dtype)


class FrequencyBlock(nn.Module):
    """Frequency filter and residual feed-forward path from M2TR."""

    def __init__(
        self, channels: int, height: int, width: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.filter = LearnableFrequencyFilter(channels, height, width)
        self.feed_forward = FeedForward2d(channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.feed_forward(self.filter(x))


class CrossModalFusion(nn.Module):
    """Query RGB features and retrieve complementary frequency artifacts.

    ``pool_size`` limits only frequency keys/values. This preserves one fused
    output per RGB location while avoiding the quadratic 56x56 attention matrix
    of the original 320px M2TR implementation.
    """

    def __init__(
        self, channels: int, hidden_channels: int, pool_size: int | None = 14
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.pool_size = pool_size
        self.query = nn.Conv2d(channels, hidden_channels, 1)
        self.key = nn.Conv2d(channels, hidden_channels, 1)
        self.value = nn.Conv2d(channels, hidden_channels, 1)
        self.scale = hidden_channels**-0.5
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, rgb: Tensor, frequency: Tensor) -> Tensor:
        if rgb.shape != frequency.shape:
            raise ValueError("RGB and frequency feature maps must have the same shape")
        batch, _, height, width = rgb.shape
        context = frequency
        if self.pool_size is not None:
            target_height = min(self.pool_size, height)
            target_width = min(self.pool_size, width)
            context = F.adaptive_avg_pool2d(context, (target_height, target_width))

        query = self.query(rgb).flatten(2).transpose(1, 2)
        key = self.key(context).flatten(2)
        value = self.value(context).flatten(2).transpose(1, 2)
        attention = ((query * self.scale) @ key).softmax(dim=-1)
        fused = (attention @ value).transpose(1, 2).reshape(
            batch, -1, height, width
        )
        return rgb + self.output(fused)


class M2TRStage(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_sizes: Sequence[int],
        feature_size: int,
        fusion_hidden_channels: int,
        fusion_pool_size: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.transformer = MultiScaleTransformerBlock(
            channels, patch_sizes, dropout=dropout
        )
        self.frequency = FrequencyBlock(
            channels, feature_size, feature_size, dropout=dropout
        )
        self.fusion = CrossModalFusion(
            channels, fusion_hidden_channels, pool_size=fusion_pool_size
        )

    def forward(self, x: Tensor) -> Tensor:
        rgb = self.transformer(x)
        frequency = self.frequency(rgb)
        return self.fusion(rgb, frequency)


class M2TRFeatureBranch(nn.Module):
    """Stacked M2TR stages adapted to FA-ViT's H/4 convolutional feature map."""

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        image_size: int = 224,
        channels: int = 64,
        depth: int = 4,
        patch_sizes: Sequence[int] | None = None,
        fusion_hidden_channels: int | None = None,
        fusion_pool_size: int | None = 14,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % 4:
            raise ValueError("image_size must be divisible by four")
        if depth <= 0:
            raise ValueError("M2TR depth must be positive")
        feature_size = image_size // 4
        if patch_sizes is None:
            if feature_size % 8:
                raise ValueError("the default M2TR scales require image_size / 4 divisible by 8")
            patch_sizes = (
                feature_size,
                feature_size // 2,
                feature_size // 4,
                feature_size // 8,
            )
        patch_sizes = tuple(int(size) for size in patch_sizes)
        if any(feature_size % size for size in patch_sizes):
            raise ValueError("every M2TR patch size must divide image_size / 4")
        hidden_channels = fusion_hidden_channels or max(channels // 2, 1)
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.stages = nn.ModuleList(
            M2TRStage(
                channels=channels,
                patch_sizes=patch_sizes,
                feature_size=feature_size,
                fusion_hidden_channels=hidden_channels,
                fusion_pool_size=fusion_pool_size,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.token_projection = nn.Conv2d(channels, embed_dim, 1)

    def forward(self, spatial_features: Tensor, token_grid: int) -> Tensor:
        x = self.input_projection(spatial_features)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool2d(x, (token_grid, token_grid))
        return self.token_projection(x).flatten(2).transpose(1, 2)

