import torch
from torch import nn

from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
from nnunetv2.training.pipeline_config import EnhancementConfig


class TinyBackbone(nn.Module):
    def __init__(self, deep_supervision=False):
        super().__init__()
        self.decoder = nn.Module()
        self.projection = nn.Conv3d(1, 2, kernel_size=1, bias=False)
        self.deep_supervision = deep_supervision
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        output = self.projection(x)
        if self.deep_supervision:
            return [output, output + 10]
        return output


def _input():
    return torch.arange(2 * 1 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8, 8)


def test_all_disabled_pipeline_is_exact_backbone_identity():
    backbone = TinyBackbone()
    pipeline = EnhancementPipeline(backbone, 1, 2, EnhancementConfig())
    x = _input()

    actual = pipeline(x)

    assert torch.equal(actual, backbone.projection(backbone.last_input))
    assert torch.equal(backbone.last_input, x)


def test_aniso_only_transforms_backbone_input():
    backbone = TinyBackbone()
    config = EnhancementConfig(enable_anisotropic_attention=True)
    pipeline = EnhancementPipeline(backbone, 1, 2, config)
    x = _input()

    pipeline(x)

    assert not torch.allclose(backbone.last_input, x)


def test_tubular_only_transforms_final_tensor_output():
    backbone = TinyBackbone()
    config = EnhancementConfig(enable_tubular_enhancement=True)
    pipeline = EnhancementPipeline(backbone, 1, 2, config)
    x = _input()

    actual = pipeline(x)
    base = backbone.projection(backbone.last_input)

    assert not torch.allclose(actual, base)


def test_aniso_and_tubular_compose_in_order():
    backbone = TinyBackbone()
    config = EnhancementConfig(
        enable_anisotropic_attention=True,
        enable_tubular_enhancement=True,
    )
    pipeline = EnhancementPipeline(backbone, 1, 2, config)
    x = _input()

    actual = pipeline(x)
    expected = pipeline.tub_enh(backbone.projection(pipeline.aniso_attn(x)))

    assert torch.allclose(actual, expected)


def test_deep_supervision_only_refines_primary_output():
    backbone = TinyBackbone(deep_supervision=True)
    config = EnhancementConfig(enable_tubular_enhancement=True)
    pipeline = EnhancementPipeline(backbone, 1, 2, config)
    x = _input()

    actual = pipeline(x)
    base = backbone.projection(backbone.last_input)

    assert isinstance(actual, list)
    assert not torch.allclose(actual[0], base)
    assert torch.equal(actual[1], base + 10)


def test_decoder_is_delegated_to_backbone():
    backbone = TinyBackbone()
    pipeline = EnhancementPipeline(backbone, 1, 2, EnhancementConfig())

    assert pipeline.decoder is backbone.decoder


def test_state_dict_keeps_legacy_module_prefixes_and_constructor():
    backbone = TinyBackbone()
    pipeline = EnhancementPipeline(
        backbone,
        1,
        2,
        EnhancementConfig(
            enable_anisotropic_attention=True,
            enable_tubular_enhancement=True,
        ),
    )
    legacy = AnisoStentUNet(TinyBackbone(), 1, 2)

    keys = pipeline.state_dict().keys()
    legacy_keys = legacy.state_dict().keys()

    assert any(key.startswith("aniso_attn.") for key in keys)
    assert any(key.startswith("backbone.") for key in keys)
    assert any(key.startswith("tub_enh.") for key in keys)
    assert any(key.startswith("aniso_attn.") for key in legacy_keys)
    assert any(key.startswith("backbone.") for key in legacy_keys)
    assert any(key.startswith("tub_enh.") for key in legacy_keys)
