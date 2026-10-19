from torch import nn

from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline


class AnisoStentUNet(EnhancementPipeline):
    """
    Wraps a standard nnU-Net backbone with:

    1. **AnisotropicAttention** (pre-processing) — suppresses directional
       streak artifacts from limited-angle back-projection.
    2. **TubularEnhancement** (post-processing) — recovers thin tubular
       structures (stents, vessels) via multi-scale dilated convolutions.

    Both modules use residual connections so the pipeline gracefully degrades
    to the base U-Net when the extra modules are not beneficial.

    Compatible with deep supervision: when the backbone returns a list of
    outputs, only the final (highest-resolution) output passes through
    TubularEnhancement; the auxiliary outputs are passed through unchanged.
    """

    def __init__(self, backbone_unet: nn.Module, in_channels: int,
                 num_classes: int):
        from nnunetv2.training.pipeline_config import EnhancementConfig

        super().__init__(
            backbone_unet,
            in_channels,
            num_classes,
            EnhancementConfig(
                enable_anisotropic_attention=True,
                enable_tubular_enhancement=True,
            ),
        )
