COMMITS
dee2041 (HEAD -> master) refactor: unify enhancement model pipeline
STAT
 nnunetv2/network_modules/__init__.py             |   1 +
 nnunetv2/network_modules/aniso_stent_unet.py     |  38 +++----
 nnunetv2/network_modules/enhancement_pipeline.py |  45 +++++++++
 nnunetv2/tests/unit/test_enhancement_pipeline.py | 120 +++++++++++++++++++++++
 4 files changed, 179 insertions(+), 25 deletions(-)
DIFF
diff --git a/nnunetv2/network_modules/__init__.py b/nnunetv2/network_modules/__init__.py
index 1452e44..9f2733f 100644
--- a/nnunetv2/network_modules/__init__.py
+++ b/nnunetv2/network_modules/__init__.py
@@ -1,4 +1,5 @@
 from .anisotropic_attention import AnisotropicAttention
 from .tubular_enhancement import TubularEnhancement
 from .projection_consistency import differentiable_forward_projection, ProjectionConsistencyLoss
+from .enhancement_pipeline import EnhancementPipeline
 from .aniso_stent_unet import AnisoStentUNet
diff --git a/nnunetv2/network_modules/aniso_stent_unet.py b/nnunetv2/network_modules/aniso_stent_unet.py
index ab47e58..57ccd32 100644
--- a/nnunetv2/network_modules/aniso_stent_unet.py
+++ b/nnunetv2/network_modules/aniso_stent_unet.py
@@ -1,47 +1,35 @@
-from typing import Union, List
-
-import torch
 from torch import nn
 
-from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
-from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
 
 
-class AnisoStentUNet(nn.Module):
+class AnisoStentUNet(EnhancementPipeline):
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
-        super().__init__()
-        self.aniso_attn = AnisotropicAttention(in_channels)
-        self.backbone = backbone_unet
-        self.tub_enh = TubularEnhancement(num_classes)
-
-    @property
-    def decoder(self):
-        """Delegate to backbone so set_deep_supervision_enabled works."""
-        return self.backbone.decoder
-
-    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
-        x = self.aniso_attn(x)
-        out = self.backbone(x)
-
-        if isinstance(out, list):
-            out[0] = self.tub_enh(out[0])
-            return out
-
-        return self.tub_enh(out)
+        from nnunetv2.training.pipeline_config import EnhancementConfig
+
+        super().__init__(
+            backbone_unet,
+            in_channels,
+            num_classes,
+            EnhancementConfig(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+            ),
+        )
diff --git a/nnunetv2/network_modules/enhancement_pipeline.py b/nnunetv2/network_modules/enhancement_pipeline.py
new file mode 100644
index 0000000..34d1ddf
--- /dev/null
+++ b/nnunetv2/network_modules/enhancement_pipeline.py
@@ -0,0 +1,45 @@
+from typing import List, Union
+
+import torch
+from torch import nn
+
+from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
+from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+class EnhancementPipeline(nn.Module):
+    """Apply the configured input and output enhancements around a backbone."""
+
+    def __init__(
+        self,
+        backbone: nn.Module,
+        in_channels: int,
+        num_classes: int,
+        config: EnhancementConfig,
+    ):
+        super().__init__()
+        self.aniso_attn = (
+            AnisotropicAttention(in_channels)
+            if config.enable_anisotropic_attention
+            else nn.Identity()
+        )
+        self.backbone = backbone
+        self.tub_enh = (
+            TubularEnhancement(num_classes)
+            if config.enable_tubular_enhancement
+            else nn.Identity()
+        )
+
+    @property
+    def decoder(self):
+        return self.backbone.decoder
+
+    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
+        out = self.backbone(self.aniso_attn(x))
+        if isinstance(out, list):
+            if not isinstance(self.tub_enh, nn.Identity):
+                out = list(out)
+                out[0] = self.tub_enh(out[0])
+            return out
+        return self.tub_enh(out)
diff --git a/nnunetv2/tests/unit/test_enhancement_pipeline.py b/nnunetv2/tests/unit/test_enhancement_pipeline.py
new file mode 100644
index 0000000..98001a0
--- /dev/null
+++ b/nnunetv2/tests/unit/test_enhancement_pipeline.py
@@ -0,0 +1,120 @@
+import torch
+from torch import nn
+
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+class TinyBackbone(nn.Module):
+    def __init__(self, deep_supervision=False):
+        super().__init__()
+        self.decoder = nn.Module()
+        self.projection = nn.Conv3d(1, 2, kernel_size=1, bias=False)
+        self.deep_supervision = deep_supervision
+        self.last_input = None
+
+    def forward(self, x):
+        self.last_input = x
+        output = self.projection(x)
+        if self.deep_supervision:
+            return [output, output + 10]
+        return output
+
+
+def _input():
+    return torch.arange(2 * 1 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8, 8)
+
+
+def test_all_disabled_pipeline_is_exact_backbone_identity():
+    backbone = TinyBackbone()
+    pipeline = EnhancementPipeline(backbone, 1, 2, EnhancementConfig())
+    x = _input()
+
+    actual = pipeline(x)
+
+    assert torch.equal(actual, backbone.projection(backbone.last_input))
+    assert torch.equal(backbone.last_input, x)
+
+
+def test_aniso_only_transforms_backbone_input():
+    backbone = TinyBackbone()
+    config = EnhancementConfig(enable_anisotropic_attention=True)
+    pipeline = EnhancementPipeline(backbone, 1, 2, config)
+    x = _input()
+
+    pipeline(x)
+
+    assert not torch.allclose(backbone.last_input, x)
+
+
+def test_tubular_only_transforms_final_tensor_output():
+    backbone = TinyBackbone()
+    config = EnhancementConfig(enable_tubular_enhancement=True)
+    pipeline = EnhancementPipeline(backbone, 1, 2, config)
+    x = _input()
+
+    actual = pipeline(x)
+    base = backbone.projection(backbone.last_input)
+
+    assert not torch.allclose(actual, base)
+
+
+def test_aniso_and_tubular_compose_in_order():
+    backbone = TinyBackbone()
+    config = EnhancementConfig(
+        enable_anisotropic_attention=True,
+        enable_tubular_enhancement=True,
+    )
+    pipeline = EnhancementPipeline(backbone, 1, 2, config)
+    x = _input()
+
+    actual = pipeline(x)
+    expected = pipeline.tub_enh(backbone.projection(pipeline.aniso_attn(x)))
+
+    assert torch.allclose(actual, expected)
+
+
+def test_deep_supervision_only_refines_primary_output():
+    backbone = TinyBackbone(deep_supervision=True)
+    config = EnhancementConfig(enable_tubular_enhancement=True)
+    pipeline = EnhancementPipeline(backbone, 1, 2, config)
+    x = _input()
+
+    actual = pipeline(x)
+    base = backbone.projection(backbone.last_input)
+
+    assert isinstance(actual, list)
+    assert not torch.allclose(actual[0], base)
+    assert torch.equal(actual[1], base + 10)
+
+
+def test_decoder_is_delegated_to_backbone():
+    backbone = TinyBackbone()
+    pipeline = EnhancementPipeline(backbone, 1, 2, EnhancementConfig())
+
+    assert pipeline.decoder is backbone.decoder
+
+
+def test_state_dict_keeps_legacy_module_prefixes_and_constructor():
+    backbone = TinyBackbone()
+    pipeline = EnhancementPipeline(
+        backbone,
+        1,
+        2,
+        EnhancementConfig(
+            enable_anisotropic_attention=True,
+            enable_tubular_enhancement=True,
+        ),
+    )
+    legacy = AnisoStentUNet(TinyBackbone(), 1, 2)
+
+    keys = pipeline.state_dict().keys()
+    legacy_keys = legacy.state_dict().keys()
+
+    assert any(key.startswith("aniso_attn.") for key in keys)
+    assert any(key.startswith("backbone.") for key in keys)
+    assert any(key.startswith("tub_enh.") for key in keys)
+    assert any(key.startswith("aniso_attn.") for key in legacy_keys)
+    assert any(key.startswith("backbone.") for key in legacy_keys)
+    assert any(key.startswith("tub_enh.") for key in legacy_keys)

