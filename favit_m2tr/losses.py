from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FineGrainedAdaptiveLoss(nn.Module):
    """Modified circle loss from equations (11)--(14) of FA-ViT.

    Each row is a fine-grained triplet consisting of the genuine-class
    prototype, a genuine frame, and its corresponding manipulated frame.
    """

    def __init__(self, scale: float = 24.0, margin: float = 0.25) -> None:
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(
        self,
        real_prototype: Tensor,
        real_features: Tensor,
        fake_features: Tensor,
    ) -> Tensor:
        if real_features.shape != fake_features.shape:
            raise ValueError("real_features and fake_features must have the same shape")
        if real_prototype.ndim == 1:
            real_prototype = real_prototype.unsqueeze(0).expand_as(real_features)
        if real_prototype.shape != real_features.shape:
            raise ValueError("prototype must be [D] or have the same shape as features")

        sp = F.cosine_similarity(real_prototype, real_features, dim=-1)
        sn = F.cosine_similarity(real_prototype, fake_features, dim=-1)
        gamma_p = F.relu(1.0 + self.margin - sp)
        gamma_n = F.relu(self.margin + sn)
        mp = 1.0 - self.margin
        mn = self.margin

        logit_p = self.scale * gamma_p * (sp - mp)
        logit_n = self.scale * gamma_n * (sn - mn)
        # softplus(logit_n - logit_p) is the stable form of equation (11).
        return F.softplus(logit_n - logit_p).mean()

