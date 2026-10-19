from typing import List, Union

import torch
from torch import nn

from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
from nnunetv2.training.pipeline_config import EnhancementConfig


class EnhancementPipeline(nn.Module):
    """Apply the configured input and output enhancements around a backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        num_classes: int,
        config: EnhancementConfig,
    ):
        super().__init__()
        self.aniso_attn = (
            AnisotropicAttention(in_channels)
            if config.enable_anisotropic_attention
            else nn.Identity()
        )
        self.backbone = backbone
        self.tub_enh = (
            TubularEnhancement(num_classes)
            if config.enable_tubular_enhancement
            else nn.Identity()
        )

    @property
    def decoder(self):
        return self.backbone.decoder

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = self.backbone(self.aniso_attn(x))
        if isinstance(out, list):
            if not isinstance(self.tub_enh, nn.Identity):
                out = list(out)
                out[0] = self.tub_enh(out[0])
            return out
        return self.tub_enh(out)
