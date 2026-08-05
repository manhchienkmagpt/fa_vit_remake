"""FA-ViT with M2TR multi-scale and frequency-domain adaptation."""

from .losses import FineGrainedAdaptiveLoss
from .m2tr import (
    CrossModalFusion,
    LearnableFrequencyFilter,
    M2TRFeatureBranch,
    MultiScalePatchAttention,
)
from .model import ForgeryAwareM2TRViT, create_favit_m2tr

__all__ = [
    "CrossModalFusion",
    "FineGrainedAdaptiveLoss",
    "ForgeryAwareM2TRViT",
    "LearnableFrequencyFilter",
    "M2TRFeatureBranch",
    "MultiScalePatchAttention",
    "create_favit_m2tr",
]
