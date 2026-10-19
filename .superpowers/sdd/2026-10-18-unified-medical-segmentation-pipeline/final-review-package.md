COMMITS
fed457a (HEAD -> master) docs: record Task 5 verification report
706c0cf test: verify unified enhancement pipeline
9b8f19f refactor: assemble legacy trainers from presets
cbbb68b fix: share normalized targets across auxiliary losses
4207073 refactor: compose segmentation auxiliary losses
dee2041 refactor: unify enhancement model pipeline
9e92e71 fix: harden unified pipeline contracts
ce1606b feat: add unified pipeline contracts
STAT
 .../task-3-report.md                               |  41 ++++
 .../task-4-report.md                               |  52 +++++
 .../task-5-report.md                               |  42 ++++
 TRAINING_INFERENCE_GUIDE.md                        |  46 +++++
 nnunetv2/network_modules/__init__.py               |   1 +
 nnunetv2/network_modules/aniso_stent_unet.py       |  38 ++--
 nnunetv2/network_modules/enhancement_pipeline.py   |  45 +++++
 nnunetv2/network_modules/projection_consistency.py |  44 ++--
 nnunetv2/tests/unit/test_composite_loss.py         | 222 +++++++++++++++++++++
 nnunetv2/tests/unit/test_enhancement_pipeline.py   | 120 +++++++++++
 nnunetv2/tests/unit/test_legacy_presets.py         | 113 +++++++++++
 nnunetv2/tests/unit/test_pipeline_contracts.py     | 209 +++++++++++++++++++
 nnunetv2/tests/unit/test_unified_pipeline_smoke.py |  73 +++++++
 nnunetv2/training/__init__.py                      |   4 +
 nnunetv2/training/auxiliary_targets.py             | 106 ++++++++++
 nnunetv2/training/loss/__init__.py                 |   3 +
 nnunetv2/training/loss/composite_loss.py           | 139 +++++++++++++
 nnunetv2/training/loss/dtm_losses.py               |  18 +-
 .../nnUNetTrainer/dual_network_trainers.py         |  25 +++
 .../training/nnUNetTrainer/nnUNetTrainerDTM.py     |  84 ++++----
 .../nnUNetTrainer/nnUNetTrainer_AnisoStent.py      |  97 ++++-----
 nnunetv2/training/pipeline_config.py               | 110 ++++++++++
 nnunetv2/training/pipeline_trainer.py              | 148 ++++++++++++++
 verify_modules.py                                  |  84 +++++++-
 24 files changed, 1711 insertions(+), 153 deletions(-)
DIFF
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
new file mode 100644
index 0000000..2f4bcf4
--- /dev/null
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-3-report.md
@@ -0,0 +1,41 @@
+# Task 3 report: composite segmentation, projection, and DTM loss
+
+## Result
+
+Implemented the Task 3 composite loss pipeline and preserved the scalar compatibility path for `CombinedSegProjLoss.forward(output, target)`.
+
+## TDD evidence
+
+- RED: `python -m pytest nnunetv2/tests/unit/test_composite_loss.py -q` failed during collection with `ModuleNotFoundError: nnunetv2.training.loss.composite_loss`, before the production module existed.
+- GREEN: the focused suite passed with `7 passed`.
+- Full unit suite: `python -m pytest nnunetv2/tests/unit -q` passed with `45 passed`.
+- `git diff --check`: passed.
+
+The direct `pytest` executable was unavailable in this environment, so the equivalent module invocation was used. The full suite reports one unrelated pytest-asyncio deprecation warning under Python 3.14.
+
+## Changed files
+
+- `nnunetv2/training/loss/composite_loss.py`: added `LossBreakdown` and `CompositeSegmentationLoss`; centralized final-resolution selection, target normalization, ignore masking, projection calculation, DTM monitor/active scheduling, and missing-auxiliary handling.
+- `nnunetv2/network_modules/projection_consistency.py`: adapted `CombinedSegProjLoss` to the composite implementation while retaining its scalar forward signature and import path.
+- `nnunetv2/training/loss/dtm_losses.py`: preserved the existing foreground DTM mean and added support for 5D foreground maps alongside the existing 6D stored-DTM layout.
+- `nnunetv2/training/loss/__init__.py`: exported the composite loss and breakdown.
+- `nnunetv2/tests/unit/test_composite_loss.py`: added focused coverage for all Task 3 behaviors.
+
+## Commit
+
+Commit: `refactor: compose segmentation auxiliary losses` (final hash is recorded in the task handoff)
+
+Planned message: `refactor: compose segmentation auxiliary losses`
+
+## Concerns
+
+- The existing `AuxiliaryTargets` contract requires DTM and validity-mask tensors to have exactly matching shapes, while the legacy DTM loader naturally stores class channels in a 6D DTM tensor and validity masks as 5D tensors. This task keeps the committed Task 1 contract unchanged; a later trainer adapter may need to normalize that legacy shape before constructing `AuxiliaryTargets`.
+- The environment does not expose a standalone `pytest` executable; verification used `python -m pytest`.
+- User edits in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` were left untouched.
+
+## Review fix: shared target normalization
+
+- RED: the regression test `test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree` failed because DTM received the raw three-channel regions target while projection received the two-channel ignore-masked target.
+- GREEN: after introducing one `_NormalizedTargets` result for the final output/target pair and reusing it for both projection and DTM, the focused suite passed with `8 passed`.
+- The segmentation loss continues to receive the original deep-supervision target tree unchanged.
+- Reviewer minor issues were intentionally left unchanged in this round.
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-4-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-4-report.md
new file mode 100644
index 0000000..665667a
--- /dev/null
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-4-report.md
@@ -0,0 +1,52 @@
+# Task 4 report: trainer assembly and legacy presets
+
+## Scope
+
+Implemented the shared trainer assembly layer and retained the legacy trainer
+names, constructor behavior, enhancement module names, DTM dataloader fields,
+and scalar `trainer.loss(output, target)` compatibility path.
+
+## TDD evidence
+
+- RED: `python -m pytest nnunetv2/tests/unit/test_legacy_presets.py -q`
+  initially failed during collection because the requested DTM A/B compatibility
+  classes and `pipeline_trainer` assembly module did not exist. The first raw
+  collection also exposed the environment's missing optional
+  `fft_conv_pytorch` dependency; the focused test uses an import-only stub so
+  the intended feature failure can be observed.
+- GREEN: the focused legacy test then passed: **18 passed**.
+- Unit suite: `python -m pytest nnunetv2/tests/unit -q` -> **64 passed**.
+- CPU smoke: the shared DTM active preset was assembled with a CPU composite
+  loss, legacy 6D DTM plus 5D validity mask, forward/backward was run, and the
+  result was finite with gradients present.
+- Static checks: `python -m compileall -q` for touched Python modules and
+  `git diff --check` completed without errors.
+
+## Changed files
+
+- `nnunetv2/training/pipeline_trainer.py`: centralized preset lookup, network
+  assembly, scalar loss adapter, auxiliary batch movement/normalization, and
+  loss breakdown helpers.
+- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py`: routed the
+  enhancement trainers through the shared pipeline/composite loss and made
+  component logging use the backpropagated breakdown.
+- `nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py`: routed DTM target
+  movement and composite loss calculation through the shared adapter while
+  retaining the existing DTM dataloader/preprocessing entry points.
+- `nnunetv2/training/nnUNetTrainer/dual_network_trainers.py`: added DTM A/B
+  compatibility names.
+- `nnunetv2/tests/unit/test_legacy_presets.py`: legacy import/preset/A-B/loss
+  callability and 6D/5D DTM normalization tests.
+
+## Commit
+
+`refactor: assemble legacy trainers from presets`
+
+## Concerns
+
+- The local environment does not provide `fft_conv_pytorch`, which is imported
+  by the nnU-Net trainer dependency chain. The focused compatibility test uses
+  an import-only stub because the dependency is not needed by these tests; a
+  full real trainer process still needs that optional package installed.
+- Existing user changes to `.gitignore` and `TRAINING_INFERENCE_GUIDE.md`, and
+  unrelated superpowers planning artifacts, were intentionally left untouched.
diff --git a/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
new file mode 100644
index 0000000..65c9a81
--- /dev/null
+++ b/.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md
@@ -0,0 +1,42 @@
+# Task 5 report: Documentation, verification, and regression cleanup
+
+## Result
+
+Task 5 is implemented directly on the existing `master` checkout. The new CPU smoke test exercises Aniso + Tubular + ProjCons + active DTM together, computes all loss components, performs backward, and checks finite values and gradients. The verifier now includes the same unified composition path and all existing proofs remain present.
+
+## TDD evidence
+
+- Initial command `pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q`: environment limitation; the `pytest` executable is not on PATH.
+- First `python -m pytest ...`: RED with an `AttributeError` because the new test called the scalar-compatible adapter `forward` while expecting a `LossBreakdown`. This was corrected in the test to call the documented `breakdown(...)` interface.
+- After that test-side correction, the smoke test passed immediately because Tasks 1–4 had already wired the required integration interfaces. No production API change was needed for the smoke path.
+- Existing verifier RED: `python verify_modules.py` reached Proof 5 and failed because it referenced the removed `CombinedSegProjLoss._target_to_onehot` helper. The verifier was minimally updated to build the equivalent one-hot target directly.
+
+## GREEN and verification evidence
+
+- `python -m pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q` → `1 passed, 1 warning`.
+- `python -m pytest nnunetv2/tests/unit -q` → `65 passed, 8 warnings`.
+- `python -m py_compile verify_modules.py nnunetv2/tests/unit/test_unified_pipeline_smoke.py nnunetv2/network_modules/enhancement_pipeline.py nnunetv2/training/auxiliary_targets.py nnunetv2/training/pipeline_config.py nnunetv2/training/pipeline_trainer.py nnunetv2/training/loss/composite_loss.py` → exit 0.
+- Public import check for the unified pipeline modules → `public imports ok`.
+- AST parse check for `verify_modules.py` and the new test → `AST parse ok`.
+- `git diff --check` → no whitespace errors; Git emitted only existing LF/CRLF conversion warnings.
+- `python verify_modules.py` → exit 0; all 8 proofs passed, including `Unified all-enabled composition`.
+
+The pytest warnings are dependency deprecations from `pytest_asyncio`, SciPy namespaces used by `batchgenerators`, and SWIG bindings; they do not fail the suite.
+
+## Changed files
+
+- `nnunetv2/tests/unit/test_unified_pipeline_smoke.py`: all-enabled CPU integration smoke test.
+- `verify_modules.py`: current target normalization compatibility fix plus unified all-enabled composition proof.
+- `TRAINING_INFERENCE_GUIDE.md`: unified `EnhancementConfig` model, legacy preset mapping, and migration example. Existing user edits in the guide were preserved.
+- `.superpowers/sdd/2026-10-18-unified-medical-segmentation-pipeline/task-5-report.md`: this report.
+
+The pre-existing `.gitignore` edit and other unrelated workspace artifacts were not changed or staged.
+
+## Environment-only limitations
+
+- The bare `pytest` executable is unavailable in this shell, so equivalent commands were run through `python -m pytest`.
+- No dependency prevented the requested final verification: the full unit suite and `verify_modules.py` both completed successfully on CPU.
+
+## Commit
+
+Implementation commit: `706c0cf` — `test: verify unified enhancement pipeline`.
diff --git a/TRAINING_INFERENCE_GUIDE.md b/TRAINING_INFERENCE_GUIDE.md
index fcb45f2..d519d4a 100644
--- a/TRAINING_INFERENCE_GUIDE.md
+++ b/TRAINING_INFERENCE_GUIDE.md
@@ -8,12 +8,13 @@
 4. [训练](#4-训练)
    - 4.1 [标准训练](#41-标准训练)
    - 4.2 [切换 Trainer](#42-训练时切换-trainer损失函数--策略)
    - 4.3 [DTM 训练](#43-dtm-训练)
    - 4.4 [Dual Containment 训练](#44-dual-containment-训练两网络联合训练)
    - 4.5 [AnisoStent 训练](#45-anisostent-训练网络增强模块)
+   - 4.6 [统一增强配置与迁移](#46-统一增强配置与迁移)
 5. [推理](#5-推理)
 6. [后处理](#6-后处理)
 7. [评估](#7-评估)
 8. [集成与寻找最佳配置](#8-集成与寻找最佳配置)
 9. [模型导出与共享](#9-模型导出与共享)
 10. [模式切换速查](#10-模式切换速查)
@@ -195,12 +196,13 @@ nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -p nnUNetPlans
 
 # 从断点继续训练
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --c
 
 # 仅验证（无需重新训练）
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val
+
 # 使用最优 checkpoint 验证
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val_best
 
 # 加载预训练权重
 nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
     -pretrained_weights /path/to/checkpoint.pth
@@ -441,12 +443,56 @@ nnUNetv2_predict_from_modelfolder \
     -m nnUNet_results/Dataset001_BrainTumor/\
 nnUNetTrainerAnisoStent_ProjCons__nnUNetPlans__3d_fullres
 ```
 
 AnisoStent 网络模块在推理时同样生效——各向异性注意力模块作用在输入上，管状结构增强模块作用在输出上。
 
+#### 4.6 统一增强配置与迁移
+
+Tasks 1–4 将网络增强和辅助损失收敛到一个 `EnhancementConfig`：
+
+| 配置项 | 作用 |
+|------|------|
+| `enable_anisotropic_attention` | 在 backbone 输入前启用 AnisotropicAttention |
+| `enable_tubular_enhancement` | 在最终分辨率输出上启用 TubularEnhancement |
+| `projection_angles_deg` / `projection_consistency_weight` | 启用 ProjectionConsistencyLoss 及其投影角度/权重 |
+| `dtm_mode` / `dtm_weight` / `dtm_active_weight` | 选择 DTM 的 `disabled`、`monitor` 或 `active` 模式及权重 |
+
+现有 trainer 名称仍然兼容，并映射到等价预设：
+
+| 旧 trainer | 统一配置等价项 |
+|------|------|
+| `nnUNetTrainerAnisoStent` | Aniso + Tubular |
+| `nnUNetTrainerAnisoStent_ProjCons` | Aniso + Tubular + ProjCons（`[0, 30, 60]`，权重 `0.1`） |
+| `nnUNetTrainerDTM` | DTM `monitor`（默认在 epoch 250 后可激活） |
+| `nnUNetTrainerDTM_GSL` | DTM `active`（权重 `1.0`，激活后系数 `0.5`） |
+
+新代码可直接构造统一网络和损失；训练循环仍可使用标量兼容的 `forward(output, target)`，需要日志明细时调用 `breakdown(...)`：
+
+```python
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss, build_network
+
+config = EnhancementConfig(
+    enable_anisotropic_attention=True,
+    enable_tubular_enhancement=True,
+    projection_angles_deg=(0.0, 30.0, 60.0),
+    projection_consistency_weight=0.1,
+    dtm_mode="active",
+    dtm_weight=1.0,
+)
+network = build_network(
+    architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
+    num_input_channels, num_output_channels, enable_deep_supervision, config,
+)
+loss = build_composite_loss(config, segmentation_loss, label_manager)
+breakdown = loss.breakdown(output, target, auxiliary_targets, epoch=epoch)
+```
+
+迁移时保留原 trainer 名称即可逐步切换：先用 `EnhancementConfig.from_legacy_preset("nnUNetTrainerAnisoStent_ProjCons")` 验证等价行为，再将配置显式传给 `build_network` 和 `build_composite_loss`。DTM 模式仍需要预先生成 DTM 文件；推理阶段不需要提供 DTM 输入。
+
 ---
 
 ## 5. 推理
 
 ### 5.1 标准推理（基于数据集）
 
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
@@ -1,16 +1,12 @@
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
@@ -23,25 +19,17 @@ class AnisoStentUNet(nn.Module):
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
diff --git a/nnunetv2/network_modules/projection_consistency.py b/nnunetv2/network_modules/projection_consistency.py
index af4dbc3..52c4e05 100644
--- a/nnunetv2/network_modules/projection_consistency.py
+++ b/nnunetv2/network_modules/projection_consistency.py
@@ -170,52 +170,32 @@ class CombinedSegProjLoss(nn.Module):
         label_manager: nnUNet LabelManager (for target → one-hot conversion).
     """
 
     def __init__(self, seg_loss: nn.Module, proj_loss: ProjectionConsistencyLoss,
                  proj_weight: float, label_manager):
         super().__init__()
+        from nnunetv2.training.loss.composite_loss import CompositeSegmentationLoss
+        from nnunetv2.training.pipeline_config import EnhancementConfig
+
         self.seg_loss = seg_loss
         self.proj_loss = proj_loss
         self.proj_weight = proj_weight
         self.label_manager = label_manager
-
-    def _target_to_onehot(self, target: torch.Tensor,
-                          num_classes: int) -> torch.Tensor:
-        """(B, 1, D, H, W) class indices → (B, C, D, H, W) one-hot."""
-        t = target.squeeze(1).long()
-        oh = F.one_hot(t, num_classes=num_classes)
-        return oh.permute(0, 4, 1, 2, 3).float()
+        angles = tuple(getattr(proj_loss, "angles_deg", ()))
+        config = EnhancementConfig(
+            projection_angles_deg=angles,
+            projection_consistency_weight=proj_weight,
+        )
+        self._composite = CompositeSegmentationLoss(
+            seg_loss, config, label_manager, projection_loss=proj_loss
+        )
 
     def forward(self, output, target) -> torch.Tensor:
         """
         Args:
             output: Tensor (B,C,D,H,W) or list of Tensors (deep supervision).
             target: Tensor or list of Tensors.
 
         Returns:
             Scalar combined loss.
         """
-        l_seg = self.seg_loss(output, target)
-
-        output_final = output[0] if isinstance(output, list) else output
-        target_final = target[0] if isinstance(target, list) else target
-
-        if self.label_manager.has_regions:
-            target_oh = target_final[:, :-1].float()
-            num_classes = target_oh.shape[1]
-            pred_vol = output_final
-            if self.label_manager.has_ignore_label:
-                mask = 1 - target_final[:, -1:]
-                pred_vol = pred_vol * mask
-        else:
-            num_classes = self.label_manager.num_segmentation_heads
-            target_oh = self._target_to_onehot(target_final, num_classes)
-            pred_vol = output_final
-            if self.label_manager.has_ignore_label:
-                mask = (target_final != self.label_manager.ignore_label).float()
-                pred_vol = pred_vol * mask
-                target_oh = target_oh * mask
-
-        # proj_loss applies softmax internally, so pass raw logits
-        l_proj = self.proj_loss(pred_vol, target_oh)
-
-        return l_seg + self.proj_weight * l_proj
+        return self._composite(output, target).total
diff --git a/nnunetv2/tests/unit/test_composite_loss.py b/nnunetv2/tests/unit/test_composite_loss.py
new file mode 100644
index 0000000..9a202ce
--- /dev/null
+++ b/nnunetv2/tests/unit/test_composite_loss.py
@@ -0,0 +1,222 @@
+from types import SimpleNamespace
+
+import pytest
+import torch
+from torch import nn
+
+from nnunetv2.network_modules.projection_consistency import (
+    CombinedSegProjLoss,
+)
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.loss.composite_loss import (
+    CompositeSegmentationLoss,
+    LossBreakdown,
+)
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+class RecordingSegmentationLoss(nn.Module):
+    def __init__(self, value=2.0):
+        super().__init__()
+        self.value = value
+        self.targets = []
+
+    def forward(self, output, target):
+        self.targets.append(target)
+        tensor = output[0] if isinstance(output, list) else output
+        return tensor.mean() * 0 + self.value
+
+
+class RecordingProjectionLoss(nn.Module):
+    def __init__(self, value=3.0):
+        super().__init__()
+        self.value = value
+        self.calls = []
+
+    def forward(self, output, target_onehot):
+        self.calls.append((output, target_onehot))
+        return output.mean() * 0 + self.value
+
+
+class RecordingDTMLoss(nn.Module):
+    def __init__(self, value=5.0):
+        super().__init__()
+        self.value = value
+        self.calls = []
+
+    def forward(self, target, output, dtm, valid_mask):
+        self.calls.append((target, output, dtm, valid_mask))
+        return output.mean() * 0 + self.value
+
+
+def label_manager(*, regions=False, ignore=None, classes=2):
+    return SimpleNamespace(
+        has_regions=regions,
+        has_ignore_label=ignore is not None,
+        ignore_label=ignore,
+        num_segmentation_heads=classes,
+    )
+
+
+def base_output():
+    return torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
+
+
+def test_segmentation_only_returns_named_breakdown_and_scalar_gradients():
+    seg = RecordingSegmentationLoss()
+    loss = CompositeSegmentationLoss(
+        seg, EnhancementConfig(), label_manager()
+    )
+
+    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))
+
+    assert isinstance(breakdown, LossBreakdown)
+    assert breakdown.total.item() == pytest.approx(2.0)
+    assert breakdown.segmentation.item() == pytest.approx(2.0)
+    assert breakdown.projection.item() == pytest.approx(0.0)
+    assert breakdown.dtm.item() == pytest.approx(0.0)
+    breakdown.total.backward()
+    assert torch.isfinite(breakdown.total)
+
+
+def test_projection_is_weighted_and_uses_final_resolution_one_hot_target():
+    proj = RecordingProjectionLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=2.0),
+        label_manager(classes=2),
+        projection_loss=proj,
+    )
+    output = [base_output(), torch.zeros((1, 2, 1, 1, 1), requires_grad=True)]
+    target = [torch.tensor([[[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]]]),
+              torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]
+
+    breakdown = loss(output, target)
+
+    assert breakdown.total.item() == pytest.approx(8.0)
+    assert len(proj.calls) == 1
+    assert proj.calls[0][0] is output[0]
+    assert proj.calls[0][1].shape == (1, 2, 2, 2, 2)
+    assert torch.equal(proj.calls[0][1][:, 1], target[0][:, 0].float())
+
+
+def test_dtm_monitor_is_reported_but_not_added_to_total():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(dtm_mode="monitor", dtm_weight=9.0),
+        label_manager(),
+        dtm_loss=dtm,
+    )
+    auxiliary = AuxiliaryTargets(
+        dtm=torch.ones((1, 1, 2, 2, 2)),
+        valid_mask=torch.ones((1, 1, 2, 2, 2)),
+    )
+
+    breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), auxiliary, epoch=0)
+
+    assert breakdown.dtm.item() == pytest.approx(5.0)
+    assert breakdown.total.item() == pytest.approx(2.0)
+    assert len(dtm.calls) == 1
+
+
+def test_dtm_active_weight_starts_after_configured_epoch():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(dtm_mode="active", dtm_weight=2.0, dtm_activate_after_epoch=3,
+                          dtm_active_weight=0.5),
+        label_manager(),
+        dtm_loss=dtm,
+    )
+    auxiliary = AuxiliaryTargets(torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2)))
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+
+    before = loss(base_output(), target, auxiliary, epoch=2)
+    after = loss(base_output(), target, auxiliary, epoch=3)
+
+    assert before.total.item() == pytest.approx(2.0)
+    assert after.total.item() == pytest.approx(2.0 + 2.0 * 0.5 * 5.0)
+
+
+def test_missing_auxiliary_data_keeps_dtm_zero_without_calling_loss():
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(), EnhancementConfig(dtm_mode="active", dtm_weight=1.0),
+        label_manager(), dtm_loss=dtm,
+    )
+
+    with pytest.warns(RuntimeWarning, match="AuxiliaryTargets are missing"):
+        breakdown = loss(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), epoch=10)
+
+    assert breakdown.dtm.item() == 0.0
+    assert len(dtm.calls) == 0
+
+
+def test_regions_and_ignore_label_are_normalized_for_projection_mask():
+    proj = RecordingProjectionLoss()
+    loss = CompositeSegmentationLoss(
+        RecordingSegmentationLoss(),
+        EnhancementConfig(projection_angles_deg=(0.0,), projection_consistency_weight=1.0),
+        label_manager(regions=True, ignore=99, classes=2), projection_loss=proj,
+    )
+    target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
+    target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
+    target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
+    target[0, 2, 0, 0, 0] = 1
+
+    loss(base_output(), target)
+
+    pred, target_onehot = proj.calls[0]
+    assert target_onehot.shape[1] == 2
+    assert target_onehot[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+    assert pred[0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+
+
+def test_dtm_receives_projection_normalized_final_target_while_segmentation_keeps_target_tree():
+    seg = RecordingSegmentationLoss()
+    proj = RecordingProjectionLoss()
+    dtm = RecordingDTMLoss()
+    loss = CompositeSegmentationLoss(
+        seg,
+        EnhancementConfig(
+            projection_angles_deg=(0.0,),
+            projection_consistency_weight=1.0,
+            dtm_mode="active",
+            dtm_weight=1.0,
+            dtm_active_weight=1.0,
+        ),
+        label_manager(regions=True, ignore=99, classes=2),
+        projection_loss=proj,
+        dtm_loss=dtm,
+    )
+    final_target = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
+    final_target[0, 0, 0] = torch.tensor([[1, 0], [0, 1]])
+    final_target[0, 1, 0] = torch.tensor([[0, 1], [1, 0]])
+    final_target[0, 2, 0, 0, 0] = 1
+    target_tree = [final_target, torch.zeros((1, 3, 1, 1, 1))]
+    auxiliary = AuxiliaryTargets(
+        torch.ones((1, 1, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2))
+    )
+
+    loss([base_output(), base_output()], target_tree, auxiliary, epoch=0)
+
+    assert seg.targets[-1] is target_tree
+    assert torch.equal(dtm.calls[0][0], proj.calls[0][1])
+    assert dtm.calls[0][0].shape == (1, 2, 2, 2, 2)
+    assert dtm.calls[0][0][0, :, 0, 0, 0].tolist() == [0.0, 0.0]
+
+
+def test_deep_supervision_segmentation_loss_receives_original_tree_and_combined_compatibility_is_scalar():
+    seg = RecordingSegmentationLoss()
+    composite = CompositeSegmentationLoss(seg, EnhancementConfig(), label_manager())
+    output = [base_output(), base_output()]
+    target = [torch.zeros((1, 1, 2, 2, 2), dtype=torch.long), torch.zeros((1, 1, 1, 1, 1), dtype=torch.long)]
+
+    result = composite(output, target)
+
+    assert seg.targets[-1] is target
+    assert result.total.ndim == 0
+    combined = CombinedSegProjLoss(seg, nn.Identity(), 0.0, label_manager())
+    scalar = combined(base_output(), torch.zeros((1, 1, 2, 2, 2), dtype=torch.long))
+    assert isinstance(scalar, torch.Tensor) and scalar.ndim == 0
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
diff --git a/nnunetv2/tests/unit/test_legacy_presets.py b/nnunetv2/tests/unit/test_legacy_presets.py
new file mode 100644
index 0000000..ae3fe31
--- /dev/null
+++ b/nnunetv2/tests/unit/test_legacy_presets.py
@@ -0,0 +1,113 @@
+from types import SimpleNamespace
+import sys
+import types
+
+import pytest
+import torch
+from torch import nn
+
+
+if "fft_conv_pytorch" not in sys.modules:
+    sys.modules["fft_conv_pytorch"] = types.SimpleNamespace(fft_conv=None)
+
+from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_AnisoStent import (
+    nnUNetTrainerAnisoStent,
+    nnUNetTrainerAnisoStent_A,
+    nnUNetTrainerAnisoStent_B,
+    nnUNetTrainerAnisoStent_ProjCons,
+    nnUNetTrainerAnisoStent_ProjCons_A,
+    nnUNetTrainerAnisoStent_ProjCons_B,
+)
+from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDTM import (
+    nnUNetTrainerDTM,
+    nnUNetTrainerDTM_GSL,
+)
+from nnunetv2.training.nnUNetTrainer.dual_network_trainers import (
+    nnUNetTrainerDTM_A,
+    nnUNetTrainerDTM_B,
+    nnUNetTrainerDTM_GSL_A,
+    nnUNetTrainerDTM_GSL_B,
+)
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import (
+    build_legacy_loss,
+    move_auxiliary_batch,
+)
+
+
+LEGACY = {
+    nnUNetTrainerAnisoStent: "nnUNetTrainerAnisoStent",
+    nnUNetTrainerAnisoStent_A: "nnUNetTrainerAnisoStent_A",
+    nnUNetTrainerAnisoStent_B: "nnUNetTrainerAnisoStent_B",
+    nnUNetTrainerAnisoStent_ProjCons: "nnUNetTrainerAnisoStent_ProjCons",
+    nnUNetTrainerAnisoStent_ProjCons_A: "nnUNetTrainerAnisoStent_ProjCons_A",
+    nnUNetTrainerAnisoStent_ProjCons_B: "nnUNetTrainerAnisoStent_ProjCons_B",
+    nnUNetTrainerDTM: "nnUNetTrainerDTM",
+    nnUNetTrainerDTM_GSL: "nnUNetTrainerDTM_GSL",
+    nnUNetTrainerDTM_A: "nnUNetTrainerDTM_A",
+    nnUNetTrainerDTM_B: "nnUNetTrainerDTM_B",
+    nnUNetTrainerDTM_GSL_A: "nnUNetTrainerDTM_GSL_A",
+    nnUNetTrainerDTM_GSL_B: "nnUNetTrainerDTM_GSL_B",
+}
+
+
+@pytest.mark.parametrize("trainer_class,preset_name", LEGACY.items())
+def test_every_legacy_trainer_exposes_its_compatibility_preset(trainer_class, preset_name):
+    assert trainer_class.enhancement_config() == EnhancementConfig.from_legacy_preset(preset_name)
+
+
+@pytest.mark.parametrize(
+    ("trainer_a", "trainer_b"),
+    [
+        (nnUNetTrainerAnisoStent_A, nnUNetTrainerAnisoStent_B),
+        (nnUNetTrainerAnisoStent_ProjCons_A, nnUNetTrainerAnisoStent_ProjCons_B),
+        (nnUNetTrainerDTM_A, nnUNetTrainerDTM_B),
+        (nnUNetTrainerDTM_GSL_A, nnUNetTrainerDTM_GSL_B),
+    ],
+)
+def test_a_b_legacy_pairs_have_equal_configs(trainer_a, trainer_b):
+    assert trainer_a.enhancement_config() == trainer_b.enhancement_config()
+
+
+class _ScalarSegmentationLoss(nn.Module):
+    def forward(self, output, target):
+        tensor = output[0] if isinstance(output, (list, tuple)) else output
+        return tensor.mean() * 0 + 1.0
+
+
+def _label_manager():
+    return SimpleNamespace(
+        has_regions=False,
+        has_ignore_label=False,
+        ignore_label=None,
+        num_segmentation_heads=2,
+    )
+
+
+def test_legacy_loss_adapter_keeps_trainer_loss_output_target_callable():
+    loss = build_legacy_loss(
+        "nnUNetTrainerAnisoStent",
+        _ScalarSegmentationLoss(),
+        _label_manager(),
+    )
+    output = torch.zeros((1, 2, 2, 2, 2), requires_grad=True)
+    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
+
+    value = loss(output, target)
+
+    assert isinstance(value, torch.Tensor)
+    assert value.ndim == 0
+    value.backward()
+    assert output.grad is not None
+
+
+def test_legacy_dtm_adapter_normalizes_6d_map_and_5d_validity_mask():
+    dtm = torch.zeros((2, 1, 4, 4, 4, 2))
+    valid_mask = torch.ones((2, 1, 4, 4, 4), dtype=torch.int16)
+
+    auxiliary = move_auxiliary_batch({"dtm": dtm, "valid_mask": valid_mask}, "cpu")
+
+    assert auxiliary.dtm.shape == dtm.shape
+    assert auxiliary.valid_mask.shape == dtm.shape
+    assert torch.equal(auxiliary.valid_mask[..., 0], valid_mask)
+    assert torch.equal(auxiliary.valid_mask[..., 1], valid_mask)
diff --git a/nnunetv2/tests/unit/test_pipeline_contracts.py b/nnunetv2/tests/unit/test_pipeline_contracts.py
new file mode 100644
index 0000000..243e342
--- /dev/null
+++ b/nnunetv2/tests/unit/test_pipeline_contracts.py
@@ -0,0 +1,209 @@
+import pytest
+import torch
+
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+def test_enhancement_config_defaults_are_disabled_and_immutable():
+    config = EnhancementConfig()
+
+    assert config.enable_anisotropic_attention is False
+    assert config.enable_tubular_enhancement is False
+    assert config.projection_angles_deg == ()
+    assert config.projection_consistency_weight == 0.0
+    assert config.dtm_mode == "disabled"
+    assert config.dtm_weight == 0.0
+    assert config.dtm_activate_after_epoch == 0
+    assert config.dtm_active_weight == 0.5
+    with pytest.raises((AttributeError, TypeError)):
+        config.dtm_mode = "active"
+
+
+@pytest.mark.parametrize(
+    ("preset", "expected"),
+    [
+        (
+            "nnUNetTrainerAnisoStent",
+            dict(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(),
+                projection_consistency_weight=0.0,
+                dtm_mode="disabled",
+            ),
+        ),
+        (
+            "nnUNetTrainerAnisoStent_ProjCons",
+            dict(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(0.0, 30.0, 60.0),
+                projection_consistency_weight=0.1,
+                dtm_mode="disabled",
+            ),
+        ),
+        (
+            "nnUNetTrainerDTM",
+            dict(
+                dtm_mode="monitor",
+                dtm_weight=0.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            ),
+        ),
+        (
+            "nnUNetTrainerDTM_GSL",
+            dict(
+                dtm_mode="active",
+                dtm_weight=1.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            ),
+        ),
+    ],
+)
+def test_legacy_presets_preserve_existing_trainer_values(preset, expected):
+    config = EnhancementConfig.from_legacy_preset(preset)
+
+    for field, value in expected.items():
+        assert getattr(config, field) == value
+
+
+def test_legacy_preset_accepts_short_names_and_rejects_unknown_names():
+    assert EnhancementConfig.from_legacy_preset("aniso_stent").enable_tubular_enhancement
+    assert EnhancementConfig.from_legacy_preset("dtm_gsl").dtm_mode == "active"
+    with pytest.raises(ValueError, match="Unknown legacy preset"):
+        EnhancementConfig.from_legacy_preset("not-a-trainer")
+
+
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"projection_consistency_weight": -0.1},
+        {"dtm_weight": -0.1},
+        {"dtm_active_weight": -0.1},
+        {"dtm_activate_after_epoch": -1},
+        {"dtm_mode": "sometimes"},
+        {"projection_consistency_weight": 0.1, "projection_angles_deg": ()},
+    ],
+)
+def test_enhancement_config_rejects_invalid_values(kwargs):
+    with pytest.raises(ValueError):
+        EnhancementConfig(**kwargs)
+
+
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"projection_angles_deg": (float("nan"),)},
+        {"projection_angles_deg": (float("inf"),)},
+        {"projection_consistency_weight": float("nan")},
+        {"dtm_weight": float("inf")},
+        {"dtm_active_weight": float("nan")},
+    ],
+)
+def test_enhancement_config_rejects_non_finite_numeric_values(kwargs):
+    with pytest.raises(ValueError, match="finite"):
+        EnhancementConfig(**kwargs)
+
+
+@pytest.mark.parametrize(
+    "kwargs",
+    [
+        {"dtm_activate_after_epoch": 1.5},
+        {"dtm_activate_after_epoch": True},
+        {"enable_anisotropic_attention": 1},
+        {"enable_tubular_enhancement": "yes"},
+    ],
+)
+def test_enhancement_config_rejects_non_integer_epochs_and_non_boolean_flags(kwargs):
+    with pytest.raises(TypeError):
+        EnhancementConfig(**kwargs)
+
+
+def test_auxiliary_targets_to_moves_nested_tensor_lists_recursively():
+    targets = AuxiliaryTargets(
+        dtm=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
+        valid_mask=[torch.ones(2, 1, 8, 8), [torch.ones(2, 1, 4, 4)]],
+    )
+
+    moved = targets.to(torch.device("meta"))
+
+    assert moved.dtm[0].device.type == "meta"
+    assert moved.dtm[1][0].device.type == "meta"
+    assert moved.valid_mask[1][0].shape == (2, 1, 4, 4)
+    assert targets.dtm[0].device.type == "cpu"
+
+
+def test_empty_auxiliary_targets_have_no_final_resolution_data():
+    targets = AuxiliaryTargets.empty()
+
+    assert targets.dtm is None
+    assert targets.valid_mask is None
+    assert targets.final_resolution() == (None, None)
+    assert targets.to("cpu") == targets
+
+
+def test_auxiliary_targets_select_first_deep_supervision_resolution():
+    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
+    valid_mask = [torch.ones(2, 1, 8, 8), torch.ones(2, 1, 4, 4)]
+
+    final_dtm, final_mask = AuxiliaryTargets(dtm, valid_mask).final_resolution()
+
+    assert final_dtm.shape == (2, 1, 8, 8)
+    assert final_mask.shape == (2, 1, 8, 8)
+
+
+@pytest.mark.parametrize(
+    ("dtm", "valid_mask", "expected_shapes"),
+    [
+        (
+            torch.zeros(2, 1, 8, 8),
+            torch.ones(3, 1, 8, 8),
+            ((2, 1, 8, 8), (3, 1, 8, 8)),
+        ),
+        (
+            [torch.zeros(2, 1, 8, 8)],
+            [torch.ones(2, 1, 4, 4)],
+            ((2, 1, 8, 8), (2, 1, 4, 4)),
+        ),
+    ],
+)
+def test_auxiliary_targets_reject_mismatched_batch_or_spatial_shapes(
+    dtm, valid_mask, expected_shapes
+):
+    with pytest.raises(ValueError) as error:
+        AuxiliaryTargets(dtm, valid_mask)
+
+    assert str(expected_shapes[0]) in str(error.value)
+    assert str(expected_shapes[1]) in str(error.value)
+
+
+def test_auxiliary_targets_list_length_error_includes_all_per_level_shapes():
+    dtm = [torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4)]
+    valid_mask = [torch.ones(2, 1, 8, 8)]
+
+    with pytest.raises(ValueError) as error:
+        AuxiliaryTargets(dtm, valid_mask)
+
+    message = str(error.value)
+    assert "dtm_shapes=[(2, 1, 8, 8), (2, 1, 4, 4)]" in message
+    assert "valid_mask_shapes=[(2, 1, 8, 8)]" in message
+
+
+@pytest.mark.parametrize(
+    ("field", "value"),
+    [
+        ("dtm", [torch.zeros(2, 1, 8, 8), "not-a-tensor"]),
+        ("valid_mask", [torch.ones(2, 1, 8, 8), {"bad": "tree"}]),
+        ("dtm", []),
+        ("valid_mask", ()),
+    ],
+)
+def test_auxiliary_targets_validate_each_non_none_tree_at_construction(field, value):
+    kwargs = {"dtm": None, "valid_mask": None}
+    kwargs[field] = value
+
+    with pytest.raises((TypeError, ValueError)):
+        AuxiliaryTargets(**kwargs)
diff --git a/nnunetv2/tests/unit/test_unified_pipeline_smoke.py b/nnunetv2/tests/unit/test_unified_pipeline_smoke.py
new file mode 100644
index 0000000..092338b
--- /dev/null
+++ b/nnunetv2/tests/unit/test_unified_pipeline_smoke.py
@@ -0,0 +1,73 @@
+from types import SimpleNamespace
+
+import torch
+from torch import nn
+
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+
+
+class TinyBackbone(nn.Module):
+    def __init__(self, in_channels=1, num_classes=2):
+        super().__init__()
+        self.decoder = nn.Module()
+        self.projection = nn.Conv3d(in_channels, num_classes, kernel_size=1)
+
+    def forward(self, x):
+        return self.projection(x)
+
+
+class TinySegmentationLoss(nn.Module):
+    def forward(self, output, target):
+        return nn.functional.cross_entropy(output, target[:, 0].long())
+
+
+def test_all_enabled_cpu_pipeline_loss_backward_is_finite():
+    torch.manual_seed(0)
+    config = EnhancementConfig(
+        enable_anisotropic_attention=True,
+        enable_tubular_enhancement=True,
+        projection_angles_deg=(0.0,),
+        projection_consistency_weight=0.25,
+        dtm_mode="active",
+        dtm_weight=0.5,
+        dtm_activate_after_epoch=0,
+        dtm_active_weight=1.0,
+    )
+    pipeline = EnhancementPipeline(TinyBackbone(), 1, 2, config).cpu()
+    label_manager = SimpleNamespace(
+        has_regions=False,
+        has_ignore_label=False,
+        ignore_label=None,
+        num_segmentation_heads=2,
+    )
+    loss = build_composite_loss(config, TinySegmentationLoss(), label_manager).cpu()
+
+    image = torch.randn(1, 1, 8, 8, 8)
+    target = torch.randint(0, 2, (1, 1, 8, 8, 8))
+    dtm = torch.rand(1, 1, 8, 8, 8, 2)
+    valid_mask = torch.ones_like(dtm)
+
+    output = pipeline(image)
+    breakdown = loss.breakdown(
+        output,
+        target,
+        AuxiliaryTargets(dtm=dtm, valid_mask=valid_mask),
+        epoch=0,
+    )
+    breakdown.total.backward()
+
+    for component in (
+        breakdown.total,
+        breakdown.segmentation,
+        breakdown.projection,
+        breakdown.dtm,
+    ):
+        assert torch.isfinite(component)
+    assert any(
+        parameter.grad is not None and torch.isfinite(parameter.grad).all()
+        for parameter in pipeline.parameters()
+        if parameter.requires_grad
+    )
diff --git a/nnunetv2/training/__init__.py b/nnunetv2/training/__init__.py
index e69de29..0853314 100644
--- a/nnunetv2/training/__init__.py
+++ b/nnunetv2/training/__init__.py
@@ -0,0 +1,4 @@
+from .auxiliary_targets import AuxiliaryTargets
+from .pipeline_config import EnhancementConfig
+
+__all__ = ["AuxiliaryTargets", "EnhancementConfig"]
diff --git a/nnunetv2/training/auxiliary_targets.py b/nnunetv2/training/auxiliary_targets.py
new file mode 100644
index 0000000..959ee4a
--- /dev/null
+++ b/nnunetv2/training/auxiliary_targets.py
@@ -0,0 +1,106 @@
+"""Contracts for optional DTM and validity-mask training targets."""
+
+from dataclasses import dataclass
+from typing import Optional, TypeAlias
+
+import torch
+from torch import Tensor
+
+
+TensorTree: TypeAlias = Tensor | list["TensorTree"] | tuple["TensorTree", ...]
+
+
+def _shape_summary(value: TensorTree) -> tuple[int, ...] | list[object]:
+    if isinstance(value, torch.Tensor):
+        return tuple(value.shape)
+    return [_shape_summary(item) for item in value]
+
+
+def _validate_tree(value: TensorTree, field_name: str, path: str = "root") -> None:
+    if isinstance(value, torch.Tensor):
+        return
+    if isinstance(value, (list, tuple)):
+        if not value:
+            raise ValueError(f"{field_name} tree at {path} cannot be empty")
+        for index, item in enumerate(value):
+            _validate_tree(item, field_name, f"{path}[{index}]")
+        return
+    raise TypeError(
+        f"{field_name} tree at {path} must contain only tensors and lists/tuples, "
+        f"got {type(value).__name__}"
+    )
+
+
+def _validate_matching_shapes(dtm: TensorTree, valid_mask: TensorTree) -> None:
+    if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
+        if tuple(dtm.shape) != tuple(valid_mask.shape):
+            raise ValueError(
+                "DTM and valid_mask shapes must match: "
+                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
+            )
+        return
+
+    if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
+        if len(dtm) != len(valid_mask):
+            raise ValueError(
+                "DTM and valid_mask list lengths must match: "
+                f"dtm={len(dtm)}, valid_mask={len(valid_mask)}, "
+                f"dtm_shapes={_shape_summary(dtm)}, "
+                f"valid_mask_shapes={_shape_summary(valid_mask)}"
+            )
+        for dtm_item, mask_item in zip(dtm, valid_mask):
+            _validate_matching_shapes(dtm_item, mask_item)
+        return
+
+    raise ValueError(
+        "DTM and valid_mask must have matching tensor/list structure: "
+        f"dtm_shapes={_shape_summary(dtm)}, valid_mask_shapes={_shape_summary(valid_mask)}"
+    )
+
+
+def _move_to(value: Optional[TensorTree], device: torch.device | str) -> Optional[TensorTree]:
+    if value is None:
+        return None
+    if isinstance(value, torch.Tensor):
+        return value.to(device)
+    if isinstance(value, list):
+        return [_move_to(item, device) for item in value]  # type: ignore[list-item]
+    if isinstance(value, tuple):
+        return tuple(_move_to(item, device) for item in value)  # type: ignore[return-value]
+    raise TypeError(f"Auxiliary targets must contain tensors, got {type(value).__name__}")
+
+
+def _first_resolution(value: Optional[TensorTree]) -> Optional[Tensor]:
+    if value is None:
+        return None
+    if isinstance(value, torch.Tensor):
+        return value
+    if not value:
+        raise ValueError("Auxiliary target resolution lists cannot be empty")
+    return _first_resolution(value[0])
+
+
+@dataclass(frozen=True)
+class AuxiliaryTargets:
+    """Optional DTM and validity-mask tensors, including deep-supervision lists."""
+
+    dtm: Optional[TensorTree] = None
+    valid_mask: Optional[TensorTree] = None
+
+    def __post_init__(self) -> None:
+        if self.dtm is not None:
+            _validate_tree(self.dtm, "dtm")
+        if self.valid_mask is not None:
+            _validate_tree(self.valid_mask, "valid_mask")
+        if self.dtm is not None and self.valid_mask is not None:
+            _validate_matching_shapes(self.dtm, self.valid_mask)
+
+    @classmethod
+    def empty(cls) -> "AuxiliaryTargets":
+        return cls()
+
+    def to(self, device: torch.device | str) -> "AuxiliaryTargets":
+        return type(self)(_move_to(self.dtm, device), _move_to(self.valid_mask, device))
+
+    def final_resolution(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
+        return _first_resolution(self.dtm), _first_resolution(self.valid_mask)
diff --git a/nnunetv2/training/loss/__init__.py b/nnunetv2/training/loss/__init__.py
index e69de29..1445ed6 100644
--- a/nnunetv2/training/loss/__init__.py
+++ b/nnunetv2/training/loss/__init__.py
@@ -0,0 +1,3 @@
+from .composite_loss import CompositeSegmentationLoss, LossBreakdown
+
+__all__ = ["CompositeSegmentationLoss", "LossBreakdown"]
diff --git a/nnunetv2/training/loss/composite_loss.py b/nnunetv2/training/loss/composite_loss.py
new file mode 100644
index 0000000..7cb2c5b
--- /dev/null
+++ b/nnunetv2/training/loss/composite_loss.py
@@ -0,0 +1,139 @@
+"""Composable segmentation, projection-consistency, and DTM losses."""
+
+from dataclasses import dataclass
+import warnings
+from typing import Optional
+
+import torch
+from torch import Tensor, nn
+from torch.nn import functional as F
+
+from nnunetv2.network_modules.projection_consistency import ProjectionConsistencyLoss
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+
+
+@dataclass(frozen=True)
+class LossBreakdown:
+    """Scalar component losses produced by :class:`CompositeSegmentationLoss`."""
+
+    total: Tensor
+    segmentation: Tensor
+    projection: Tensor
+    dtm: Tensor
+
+    def as_scalar(self) -> Tensor:
+        return self.total
+
+
+@dataclass(frozen=True)
+class _NormalizedTargets:
+    output: Tensor
+    target: Tensor
+
+
+def _first_resolution(value):
+    return value[0] if isinstance(value, (list, tuple)) else value
+
+
+class CompositeSegmentationLoss(nn.Module):
+    """Compose the existing segmentation, projection, and DTM loss modules."""
+
+    def __init__(
+        self,
+        segmentation_loss: nn.Module,
+        config: EnhancementConfig,
+        label_manager,
+        projection_loss: Optional[nn.Module] = None,
+        dtm_loss: Optional[nn.Module] = None,
+    ):
+        super().__init__()
+        self.segmentation_loss = segmentation_loss
+        self.config = config
+        self.label_manager = label_manager
+        self.projection_loss = projection_loss
+        self.dtm_loss = dtm_loss
+        self._missing_auxiliary_warned = False
+
+        if config.projection_consistency_weight and projection_loss is None:
+            raise ValueError(
+                "projection_loss is required when projection consistency is enabled"
+            )
+
+    @staticmethod
+    def _as_scalar(value: Tensor | float | int, reference: Tensor) -> Tensor:
+        if isinstance(value, Tensor):
+            return value if value.ndim == 0 else value.mean()
+        return reference.new_tensor(value)
+
+    def _normalize_final_targets(self, output: Tensor, target: Tensor) -> _NormalizedTargets:
+        if self.label_manager.has_regions:
+            num_classes = self.label_manager.num_segmentation_heads
+            target_onehot = target[:, :num_classes].float()
+            valid = None
+            if self.label_manager.has_ignore_label:
+                valid = 1.0 - target[:, -1:].float()
+        else:
+            num_classes = self.label_manager.num_segmentation_heads
+            valid = None
+            target_indices = target[:, 0].long()
+            if self.label_manager.has_ignore_label:
+                valid = (target_indices != self.label_manager.ignore_label).unsqueeze(1).float()
+                target_indices = target_indices.masked_fill(valid[:, 0] == 0, 0)
+            target_onehot = F.one_hot(target_indices, num_classes=num_classes)
+            target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
+
+        if valid is not None:
+            output = output * valid
+            target_onehot = target_onehot * valid
+        return _NormalizedTargets(output=output, target=target_onehot)
+
+    def _dtm_component(
+        self, output: Tensor, target: Tensor, auxiliary: Optional[AuxiliaryTargets], epoch: Optional[int]
+    ) -> Tensor:
+        zero = output.sum() * 0
+        if self.config.dtm_mode == "disabled":
+            return zero
+        if auxiliary is None or self.dtm_loss is None:
+            if not self._missing_auxiliary_warned:
+                warnings.warn(
+                    "DTM loss is enabled but AuxiliaryTargets are missing; DTM component is zero.",
+                    RuntimeWarning,
+                    stacklevel=2,
+                )
+                self._missing_auxiliary_warned = True
+            return zero
+
+        dtm, valid_mask = auxiliary.final_resolution()
+        if dtm is None or valid_mask is None:
+            return zero
+        return self._as_scalar(self.dtm_loss(target, output, dtm, valid_mask), output)
+
+    def forward(
+        self,
+        output,
+        target,
+        auxiliary: Optional[AuxiliaryTargets] = None,
+        epoch: Optional[int] = None,
+    ) -> LossBreakdown:
+        segmentation = self._as_scalar(self.segmentation_loss(output, target), output[0] if isinstance(output, (list, tuple)) else output)
+        output_final = _first_resolution(output)
+        target_final = _first_resolution(target)
+
+        projection = output_final.sum() * 0
+        normalized = None
+        if self.config.projection_consistency_weight or self.config.dtm_mode != "disabled":
+            normalized = self._normalize_final_targets(output_final, target_final)
+
+        if self.config.projection_consistency_weight and self.projection_loss is not None:
+            projection_input, target_onehot = normalized.output, normalized.target
+            projection = self._as_scalar(self.projection_loss(projection_input, target_onehot), output_final)
+
+        normalized_output = normalized.output if normalized is not None else output_final
+        normalized_target = normalized.target if normalized is not None else target_final
+        dtm = self._dtm_component(normalized_output, normalized_target, auxiliary, epoch)
+        total = segmentation + self.config.projection_consistency_weight * projection
+        if self.config.dtm_mode == "active" and auxiliary is not None and self.dtm_loss is not None:
+            if epoch is not None and epoch >= self.config.dtm_activate_after_epoch:
+                total = total + self.config.dtm_weight * self.config.dtm_active_weight * dtm
+        return LossBreakdown(total=total, segmentation=segmentation, projection=projection, dtm=dtm)
diff --git a/nnunetv2/training/loss/dtm_losses.py b/nnunetv2/training/loss/dtm_losses.py
index 6df0844..a80ef5a 100644
--- a/nnunetv2/training/loss/dtm_losses.py
+++ b/nnunetv2/training/loss/dtm_losses.py
@@ -22,19 +22,27 @@ class GenSurfLoss(nn.Module):
         """
         y_true: (B, 1, D, H, W)
         y_pred: (B, 2, D, H, W)
         valid_mask: (B, 1, D, H, W)
         dtm: (B, channel, D, H, W, class)
         """
-        # Reshape dtm from (B, Channel, D, H, W, Class) -> (B, Class, D, H, W)
-        dtm = torch.transpose(dtm, dim0=5, dim1=1)
-        dtm = torch.squeeze(dtm, dim=5)
+        # Reshape the stored DTM from (B, Channel, D, H, W, Class) to
+        # (B, Class, D, H, W). AuxiliaryTargets may also provide the already
+        # selected foreground map as (B, 1, D, H, W).
+        if dtm.ndim == 6:
+            dtm = torch.transpose(dtm, dim0=5, dim1=1)
+            dtm = torch.squeeze(dtm, dim=5)
+        elif dtm.ndim != 5:
+            raise ValueError(f"DTM must have 5 or 6 dimensions, got {tuple(dtm.shape)}")
 
         y_pred = softmax(y_pred, dim=1)
-        # Return mean of dtm * pred for the foreground class
-        num = torch.mean(dtm[:, 1, ...] * y_pred[:, 1, ...])
+        # Return mean of dtm * pred for the foreground class. Keep the
+        # historical unmasked mean; valid_mask remains part of the signature
+        # for compatibility with existing DTM trainers.
+        dtm_foreground = dtm[:, 1, ...] if dtm.shape[1] > 1 else dtm[:, 0, ...]
+        num = torch.mean(dtm_foreground * y_pred[:, 1, ...])
         return torch.mean(num)
 
 
 class LinearSchedule:
     def __init__(self, num_epochs, init_pause=0):
         if num_epochs <= init_pause:
diff --git a/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py b/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
index 30b4016..e617520 100644
--- a/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
+++ b/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
@@ -1,9 +1,10 @@
 import torch
 
 from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
+from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDTM import nnUNetTrainerDTM, nnUNetTrainerDTM_GSL
 
 
 class nnUNetTrainerA(nnUNetTrainer):
     """
     Trainer class for network A.
     Kept separate so nnU-Net writes A and B checkpoints into different result folders.
@@ -30,6 +31,30 @@ class nnUNetTrainerB(nnUNetTrainer):
         configuration: str,
         fold: int,
         dataset_json: dict,
         device: torch.device = torch.device("cuda"),
     ):
         super().__init__(plans, configuration, fold, dataset_json, device)
+
+
+class nnUNetTrainerDTM_A(nnUNetTrainerDTM):
+    """Dual-training A compatibility name for the DTM monitor preset."""
+
+    pass
+
+
+class nnUNetTrainerDTM_B(nnUNetTrainerDTM):
+    """Dual-training B compatibility name for the DTM monitor preset."""
+
+    pass
+
+
+class nnUNetTrainerDTM_GSL_A(nnUNetTrainerDTM_GSL):
+    """Dual-training A compatibility name for the active DTM preset."""
+
+    pass
+
+
+class nnUNetTrainerDTM_GSL_B(nnUNetTrainerDTM_GSL):
+    """Dual-training B compatibility name for the active DTM preset."""
+
+    pass
diff --git a/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py b/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py
index 2d9cabd..a4d4821 100644
--- a/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py
+++ b/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDTM.py
@@ -18,20 +18,28 @@ from nnunetv2.training.dataloading.nnunet_dataset_dtm import nnUNetDatasetBlosc2
 from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
 from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
 from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
 from nnunetv2.training.loss.dtm_losses import GenSurfLoss, CombinedLoss, LinearSchedule
 from nnunetv2.training.loss.dtm_deep_supervision import DeepSupervisionWrapperDTM
 from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
+from nnunetv2.training.pipeline_trainer import (
+    LegacyTrainerAssemblyMixin,
+    build_composite_loss,
+    build_network,
+    move_auxiliary_batch,
+)
 from nnunetv2.utilities.collate_outputs import collate_outputs
 from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
 from nnunetv2.utilities.helpers import dummy_context
 
 
-class nnUNetTrainerDTM(nnUNetTrainer):
+class nnUNetTrainerDTM(LegacyTrainerAssemblyMixin, nnUNetTrainer):
     """DTM-aware trainer: tracks GSL loss but only uses DC_CE for backprop (monitoring mode)."""
 
+    enhancement_preset = "nnUNetTrainerDTM"
+
     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                  device: torch.device = torch.device('cuda')):
         super().__init__(plans, configuration, fold, dataset_json, device)
         self.gsl_alpha = 0.5
         self.gsl_activate_after_epoch = 250
 
@@ -86,19 +94,20 @@ class nnUNetTrainerDTM(nnUNetTrainer):
             self._set_batch_size_and_oversample()
 
             from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                    self.dataset_json)
 
-            self.network = self.build_network_architecture(
+            self.network = build_network(
                 self.configuration_manager.network_arch_class_name,
                 self.configuration_manager.network_arch_init_kwargs,
                 self.configuration_manager.network_arch_init_kwargs_req_import,
                 self.num_input_channels,
                 self.label_manager.num_segmentation_heads,
-                self.enable_deep_supervision
+                self.enable_deep_supervision,
+                self.enhancement_config(),
             ).to(self.device)
 
             if self._do_i_compile():
                 self.print_to_log_file('Using torch.compile...')
                 self.network = torch.compile(self.network)
 
@@ -106,20 +115,24 @@ class nnUNetTrainerDTM(nnUNetTrainer):
             if self.is_ddp:
                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                 self.network = DDP(self.network, device_ids=[self.local_rank])
 
             loss_dc_ce, loss_gsl_raw, linear_schedule = self._build_loss()
 
-            # Wrap dc_ce with deep supervision if needed
             if self.enable_deep_supervision:
                 weights = self._get_deep_supervision_weights()
-                self.loss = DeepSupervisionWrapper(loss_dc_ce, weights)
+                loss_dc_ce = DeepSupervisionWrapper(loss_dc_ce, weights)
                 self.loss_temp = DeepSupervisionWrapper(loss_gsl_raw, weights)
             else:
-                self.loss = loss_dc_ce
                 self.loss_temp = loss_gsl_raw
+            self.loss = build_composite_loss(
+                self.enhancement_config(),
+                loss_dc_ce,
+                self.label_manager,
+                dtm_loss=loss_gsl_raw,
+            )
             self.linear_schedule = linear_schedule
 
             self.dataset_class = nnUNetDatasetBlosc2DTM
 
             self.was_initialized = True
         else:
@@ -182,23 +195,16 @@ class nnUNetTrainerDTM(nnUNetTrainer):
                                                       wait_time=0.002)
         _ = next(mt_gen_train)
         _ = next(mt_gen_val)
         return mt_gen_train, mt_gen_val
 
     def _move_dtm_to_device(self, dtm, valid_mask):
-        if isinstance(dtm, list):
-            dtm = [d.to(self.device, non_blocking=True) for d in dtm]
-        elif dtm is not None:
-            dtm = dtm.to(self.device, non_blocking=True)
-
-        if isinstance(valid_mask, list):
-            valid_mask = [v.to(self.device, non_blocking=True) for v in valid_mask]
-        elif valid_mask is not None:
-            valid_mask = valid_mask.to(self.device, non_blocking=True)
-
-        return dtm, valid_mask
+        auxiliary = move_auxiliary_batch(
+            {"dtm": dtm, "valid_mask": valid_mask}, self.device
+        )
+        return auxiliary.dtm, auxiliary.valid_mask
 
     def train_step(self, batch: dict) -> dict:
         data = batch['data']
         target = batch['target']
         dtm = batch.get('dtm')
         valid_mask = batch.get('valid_mask')
@@ -206,25 +212,23 @@ class nnUNetTrainerDTM(nnUNetTrainer):
         data = data.to(self.device, non_blocking=True)
         if isinstance(target, list):
             target = [i.to(self.device, non_blocking=True) for i in target]
         else:
             target = target.to(self.device, non_blocking=True)
 
-        dtm, valid_mask = self._move_dtm_to_device(dtm, valid_mask)
+        auxiliary = move_auxiliary_batch(
+            {"dtm": dtm, "valid_mask": valid_mask}, self.device
+        )
+        dtm, valid_mask = auxiliary.dtm, auxiliary.valid_mask
 
         self.optimizer.zero_grad(set_to_none=True)
         with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
             output = self.network(data)
-            l = self.loss(output, target)
-
-            if dtm is not None and valid_mask is not None:
-                l_gsl = self.loss_temp(target, output if not isinstance(output, list) else output[0],
-                                       dtm if not isinstance(dtm, list) else dtm[0],
-                                       valid_mask if not isinstance(valid_mask, list) else valid_mask[0])
-            else:
-                l_gsl = torch.tensor(0.0)
+            breakdown = self.loss.breakdown(output, target, auxiliary, self.current_epoch)
+            l = breakdown.total
+            l_gsl = breakdown.dtm
 
         if self.grad_scaler is not None:
             self.grad_scaler.scale(l).backward()
             self.grad_scaler.unscale_(self.optimizer)
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
             self.grad_scaler.step(self.optimizer)
@@ -245,25 +249,23 @@ class nnUNetTrainerDTM(nnUNetTrainer):
         data = data.to(self.device, non_blocking=True)
         if isinstance(target, list):
             target = [i.to(self.device, non_blocking=True) for i in target]
         else:
             target = target.to(self.device, non_blocking=True)
 
-        dtm, valid_mask = self._move_dtm_to_device(dtm, valid_mask)
+        auxiliary = move_auxiliary_batch(
+            {"dtm": dtm, "valid_mask": valid_mask}, self.device
+        )
+        dtm, valid_mask = auxiliary.dtm, auxiliary.valid_mask
 
         with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
             output = self.network(data)
             del data
-            l = self.loss(output, target)
-
-            if dtm is not None and valid_mask is not None:
-                l_gsl = self.loss_temp(target, output if not isinstance(output, list) else output[0],
-                                       dtm if not isinstance(dtm, list) else dtm[0],
-                                       valid_mask if not isinstance(valid_mask, list) else valid_mask[0])
-            else:
-                l_gsl = torch.tensor(0.0)
+            breakdown = self.loss.breakdown(output, target, auxiliary, self.current_epoch)
+            l = breakdown.total
+            l_gsl = breakdown.dtm
 
         if self.enable_deep_supervision:
             output = output[0]
             target = target[0]
 
         axes = [0] + list(range(2, output.ndim))
@@ -343,15 +345,17 @@ class nnUNetTrainerDTM(nnUNetTrainer):
         if self.local_rank == 0:
             self.logger.plot_progress_png(self.output_folder)
 
         self.current_epoch += 1
 
 
-class nnUNetTrainerDTM_GSL(nnUNetTrainerDTM):
+class _LegacyDTM_GSLTrainer(nnUNetTrainerDTM):
     """DTM trainer where GSL actually participates in backpropagation via CombinedLoss."""
 
+    enhancement_preset = "nnUNetTrainerDTM_GSL"
+
     def initialize(self):
         if not self.was_initialized:
             self._set_batch_size_and_oversample()
 
             from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
@@ -439,6 +443,14 @@ class nnUNetTrainerDTM_GSL(nnUNetTrainerDTM):
             final_loss.backward()
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
             self.optimizer.step()
 
         return {'loss': final_loss.detach().cpu().numpy(),
                 'gsl_loss': gsl_loss.detach().cpu().numpy() if isinstance(gsl_loss, torch.Tensor) else gsl_loss}
+
+
+# The public name is intentionally rebound to the same shared implementation
+# as the monitor trainer. The preset controls whether DTM contributes to total.
+class nnUNetTrainerDTM_GSL(nnUNetTrainerDTM):
+    """DTM trainer where GSL participates through the shared composite loss."""
+
+    enhancement_preset = "nnUNetTrainerDTM_GSL"
diff --git a/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py b/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py
index bb757e5..6ad0058 100644
--- a/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py
+++ b/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py
@@ -1,53 +1,61 @@
 from typing import List, Union
 
 import numpy as np
 import torch
 from torch import autocast, nn
 
-from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
 from nnunetv2.network_modules.projection_consistency import (
     ProjectionConsistencyLoss, CombinedSegProjLoss,
 )
+from nnunetv2.training.pipeline_trainer import (
+    LegacyTrainerAssemblyMixin,
+    build_composite_loss,
+    build_network,
+)
 from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
 from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
 from nnunetv2.utilities.helpers import dummy_context
 from nnunetv2.utilities.collate_outputs import collate_outputs
 from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
 
 
-class nnUNetTrainerAnisoStent(nnUNetTrainer):
+class nnUNetTrainerAnisoStent(LegacyTrainerAssemblyMixin, nnUNetTrainer):
     """
     Trainer that wraps the standard nnU-Net backbone with:
 
     - **AnisotropicAttention**: suppresses directional streak artifacts from
       limited-angle back-projection.
     - **TubularEnhancement**: recovers thin tubular structures via multi-scale
       dilated convolutions.
 
     Usage:
         nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerAnisoStent
     """
 
-    @staticmethod
-    def build_network_architecture(architecture_class_name: str,
+    enhancement_preset = "nnUNetTrainerAnisoStent"
+
+    @classmethod
+    def build_network_architecture(cls, architecture_class_name: str,
                                    arch_init_kwargs: dict,
                                    arch_init_kwargs_req_import: Union[List[str], tuple],
                                    num_input_channels: int,
                                    num_output_channels: int,
                                    enable_deep_supervision: bool = True) -> nn.Module:
-        backbone = get_network_from_plans(
+        return build_network(
             architecture_class_name,
             arch_init_kwargs,
             arch_init_kwargs_req_import,
             num_input_channels,
             num_output_channels,
-            allow_init=True,
-            deep_supervision=enable_deep_supervision,
+            enable_deep_supervision,
+            cls.enhancement_config(),
         )
-        return AnisoStentUNet(backbone, num_input_channels, num_output_channels)
+
+    def _build_loss(self):
+        return build_composite_loss(self.enhancement_config(), super()._build_loss(), self.label_manager)
 
 
 class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
     """
     Extends ``nnUNetTrainerAnisoStent`` with a **projection consistency loss**.
 
@@ -60,12 +68,14 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
         - ``proj_cons_weight``: weight of the projection consistency term (default 0.1)
 
     Usage:
         nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerAnisoStent_ProjCons
     """
 
+    enhancement_preset = "nnUNetTrainerAnisoStent_ProjCons"
+
     def __init__(self, plans: dict, configuration: str, fold: int,
                  dataset_json: dict, device: torch.device = torch.device('cuda')):
         super().__init__(plans, configuration, fold, dataset_json, device)
         self.projection_angles_deg = [0, 30, 60]
         self.proj_cons_weight = 0.1
         self.logger.my_fantastic_logging['proj_losses_train'] = list()
@@ -88,12 +98,25 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
                                   ignore_label=self.label_manager.ignore_label,
                                   dice_class=MemoryEfficientSoftDiceLoss)
         if self._do_i_compile():
             loss.dc = torch.compile(loss.dc)
         return loss
 
+    def _build_loss(self):
+        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
+
+        seg_loss = self._build_seg_loss()
+        if self.enable_deep_supervision:
+            seg_loss = DeepSupervisionWrapper(seg_loss, self._get_deep_supervision_weights())
+        return build_composite_loss(
+            self.enhancement_config(),
+            seg_loss,
+            self.label_manager,
+            projection_loss=self._build_proj_cons_loss(),
+        )
+
     def _get_deep_supervision_weights(self):
         deep_supervision_scales = self._get_deep_supervision_scales()
         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
         if self.is_ddp and not self._do_i_compile():
             weights[-1] = 1e-6
         else:
@@ -141,17 +164,17 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
             if self.enable_deep_supervision:
                 weights = self._get_deep_supervision_weights()
                 seg_loss = DeepSupervisionWrapper(seg_loss, weights)
 
             proj_loss = self._build_proj_cons_loss()
 
-            self.loss = CombinedSegProjLoss(
-                seg_loss=seg_loss,
-                proj_loss=proj_loss,
-                proj_weight=self.proj_cons_weight,
-                label_manager=self.label_manager,
+            self.loss = build_composite_loss(
+                self.enhancement_config(),
+                seg_loss,
+                self.label_manager,
+                projection_loss=proj_loss,
             )
 
             self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
 
             self.was_initialized = True
         else:
@@ -167,33 +190,16 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
         else:
             target = target.to(self.device, non_blocking=True)
 
         self.optimizer.zero_grad(set_to_none=True)
         with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
             output = self.network(data)
-            l_total = self.loss(output, target)
-
-            # Extract per-component losses for logging
-            with torch.no_grad():
-                l_seg = self.loss.seg_loss(output, target)
-                output_final = output[0] if isinstance(output, list) else output
-                target_final = target[0] if isinstance(target, list) else target
-                if self.label_manager.has_regions:
-                    t_oh = target_final[:, :-1].float()
-                    pred_vol = output_final
-                    if self.label_manager.has_ignore_label:
-                        pred_vol = pred_vol * (1 - target_final[:, -1:])
-                else:
-                    t_oh = self.loss._target_to_onehot(
-                        target_final, self.label_manager.num_segmentation_heads)
-                    pred_vol = output_final
-                    if self.label_manager.has_ignore_label:
-                        m = (target_final != self.label_manager.ignore_label).float()
-                        pred_vol = pred_vol * m
-                        t_oh = t_oh * m
-                l_proj = self.loss.proj_loss(pred_vol, t_oh)
+            breakdown = self.loss.breakdown(output, target)
+            l_total = breakdown.total
+            l_seg = breakdown.segmentation
+            l_proj = breakdown.projection
 
         if self.grad_scaler is not None:
             self.grad_scaler.scale(l_total).backward()
             self.grad_scaler.unscale_(self.optimizer)
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
             self.grad_scaler.step(self.optimizer)
@@ -219,31 +225,16 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
         else:
             target = target.to(self.device, non_blocking=True)
 
         with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
             output = self.network(data)
             del data
-            l_total = self.loss(output, target)
-
-            l_seg = self.loss.seg_loss(output, target)
-            output_final = output[0] if isinstance(output, list) else output
-            target_final = target[0] if isinstance(target, list) else target
-            if self.label_manager.has_regions:
-                t_oh = target_final[:, :-1].float()
-                pred_vol = output_final
-                if self.label_manager.has_ignore_label:
-                    pred_vol = pred_vol * (1 - target_final[:, -1:])
-            else:
-                t_oh = self.loss._target_to_onehot(
-                    target_final, self.label_manager.num_segmentation_heads)
-                pred_vol = output_final
-                if self.label_manager.has_ignore_label:
-                    m = (target_final != self.label_manager.ignore_label).float()
-                    pred_vol = pred_vol * m
-                    t_oh = t_oh * m
-            l_proj = self.loss.proj_loss(pred_vol, t_oh)
+            breakdown = self.loss.breakdown(output, target)
+            l_total = breakdown.total
+            l_seg = breakdown.segmentation
+            l_proj = breakdown.projection
 
         # Dice metric computation on final output
         if self.enable_deep_supervision:
             output_metric = output[0]
             target_metric = target[0] if isinstance(target, list) else target
         else:
diff --git a/nnunetv2/training/pipeline_config.py b/nnunetv2/training/pipeline_config.py
new file mode 100644
index 0000000..7a3606f
--- /dev/null
+++ b/nnunetv2/training/pipeline_config.py
@@ -0,0 +1,110 @@
+"""Configuration contracts for the unified enhancement pipeline."""
+
+from dataclasses import dataclass
+import math
+from numbers import Real
+from typing import Literal
+
+
+DTMMode = Literal["disabled", "monitor", "active"]
+
+
+def _finite_float(name: str, value: object) -> float:
+    if isinstance(value, bool) or not isinstance(value, Real):
+        raise TypeError(f"{name} must be a real number")
+    result = float(value)
+    if not math.isfinite(result):
+        raise ValueError(f"{name} must be finite")
+    return result
+
+
+@dataclass(frozen=True)
+class EnhancementConfig:
+    """Immutable, validated configuration shared by pipeline components."""
+
+    enable_anisotropic_attention: bool = False
+    enable_tubular_enhancement: bool = False
+    projection_angles_deg: tuple[float, ...] = ()
+    projection_consistency_weight: float = 0.0
+    dtm_mode: DTMMode = "disabled"
+    dtm_weight: float = 0.0
+    dtm_activate_after_epoch: int = 0
+    dtm_active_weight: float = 0.5
+
+    def __post_init__(self) -> None:
+        if not isinstance(self.enable_anisotropic_attention, bool):
+            raise TypeError("enable_anisotropic_attention must be a bool")
+        if not isinstance(self.enable_tubular_enhancement, bool):
+            raise TypeError("enable_tubular_enhancement must be a bool")
+        if not isinstance(self.dtm_activate_after_epoch, int) or isinstance(
+            self.dtm_activate_after_epoch, bool
+        ):
+            raise TypeError("dtm_activate_after_epoch must be an integer")
+
+        angles = tuple(_finite_float("projection_angles_deg", angle) for angle in self.projection_angles_deg)
+        object.__setattr__(self, "projection_angles_deg", angles)
+        projection_weight = _finite_float(
+            "projection_consistency_weight", self.projection_consistency_weight
+        )
+        dtm_weight = _finite_float("dtm_weight", self.dtm_weight)
+        active_weight = _finite_float("dtm_active_weight", self.dtm_active_weight)
+        object.__setattr__(self, "projection_consistency_weight", projection_weight)
+        object.__setattr__(self, "dtm_weight", dtm_weight)
+        object.__setattr__(self, "dtm_active_weight", active_weight)
+
+        if self.dtm_mode not in ("disabled", "monitor", "active"):
+            raise ValueError(
+                f"dtm_mode must be 'disabled', 'monitor', or 'active', got {self.dtm_mode!r}"
+            )
+        if projection_weight < 0:
+            raise ValueError("projection_consistency_weight must be non-negative")
+        if dtm_weight < 0:
+            raise ValueError("dtm_weight must be non-negative")
+        if active_weight < 0:
+            raise ValueError("dtm_active_weight must be non-negative")
+        if self.dtm_activate_after_epoch < 0:
+            raise ValueError("dtm_activate_after_epoch must be non-negative")
+        if not angles and self.projection_consistency_weight != 0:
+            raise ValueError(
+                "projection_angles_deg cannot be empty when projection_consistency_weight is non-zero"
+            )
+
+    @classmethod
+    def from_legacy_preset(cls, name: str) -> "EnhancementConfig":
+        """Return the configuration represented by an existing trainer name."""
+
+        normalized = name.removesuffix("_A").removesuffix("_B")
+        aliases = {
+            "aniso_stent": "nnUNetTrainerAnisoStent",
+            "aniso_stent_projcons": "nnUNetTrainerAnisoStent_ProjCons",
+            "dtm": "nnUNetTrainerDTM",
+            "dtm_gsl": "nnUNetTrainerDTM_GSL",
+        }
+        normalized = aliases.get(normalized, normalized)
+
+        if normalized == "nnUNetTrainerAnisoStent":
+            return cls(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+            )
+        if normalized == "nnUNetTrainerAnisoStent_ProjCons":
+            return cls(
+                enable_anisotropic_attention=True,
+                enable_tubular_enhancement=True,
+                projection_angles_deg=(0.0, 30.0, 60.0),
+                projection_consistency_weight=0.1,
+            )
+        if normalized == "nnUNetTrainerDTM":
+            return cls(
+                dtm_mode="monitor",
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            )
+        if normalized == "nnUNetTrainerDTM_GSL":
+            return cls(
+                dtm_mode="active",
+                dtm_weight=1.0,
+                dtm_activate_after_epoch=250,
+                dtm_active_weight=0.5,
+            )
+        raise ValueError(f"Unknown legacy preset: {name!r}")
diff --git a/nnunetv2/training/pipeline_trainer.py b/nnunetv2/training/pipeline_trainer.py
new file mode 100644
index 0000000..75e7462
--- /dev/null
+++ b/nnunetv2/training/pipeline_trainer.py
@@ -0,0 +1,148 @@
+"""Shared assembly and compatibility adapters for legacy trainers."""
+
+from __future__ import annotations
+
+from typing import Any, Optional
+
+import torch
+from torch import Tensor, nn
+
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+from nnunetv2.network_modules.projection_consistency import ProjectionConsistencyLoss
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets, TensorTree
+from nnunetv2.training.loss.composite_loss import CompositeSegmentationLoss, LossBreakdown
+from nnunetv2.training.loss.dtm_losses import GenSurfLoss
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
+
+
+def legacy_config(name: str) -> EnhancementConfig:
+    return EnhancementConfig.from_legacy_preset(name)
+
+
+def _preset_name_for_class(name: str) -> str:
+    return name.removesuffix("_A").removesuffix("_B")
+
+
+class LegacyTrainerAssemblyMixin:
+    """Small class-level compatibility surface shared by legacy trainers."""
+
+    enhancement_preset: str = "nnUNetTrainer"
+
+    @classmethod
+    def enhancement_config(cls) -> EnhancementConfig:
+        return legacy_config(_preset_name_for_class(cls.enhancement_preset))
+
+    @classmethod
+    def legacy_preset_name(cls) -> str:
+        return cls.enhancement_preset
+
+
+def build_network(
+    architecture_class_name: str,
+    arch_init_kwargs: dict,
+    arch_init_kwargs_req_import,
+    num_input_channels: int,
+    num_output_channels: int,
+    enable_deep_supervision: bool,
+    config: EnhancementConfig,
+) -> nn.Module:
+    backbone = get_network_from_plans(
+        architecture_class_name,
+        arch_init_kwargs,
+        arch_init_kwargs_req_import,
+        num_input_channels,
+        num_output_channels,
+        allow_init=True,
+        deep_supervision=enable_deep_supervision,
+    )
+    if not (config.enable_anisotropic_attention or config.enable_tubular_enhancement):
+        return backbone
+    return EnhancementPipeline(backbone, num_input_channels, num_output_channels, config)
+
+
+class TrainerLossAdapter(nn.Module):
+    """Scalar-compatible trainer loss retaining the named composite breakdown."""
+
+    def __init__(self, composite: CompositeSegmentationLoss):
+        super().__init__()
+        self.composite = composite
+        self.seg_loss = composite.segmentation_loss
+        self.proj_loss = composite.projection_loss
+        self.dtm_loss = composite.dtm_loss
+        self.last_breakdown: Optional[LossBreakdown] = None
+
+    def breakdown(self, output, target, auxiliary=None, epoch=None) -> LossBreakdown:
+        self.last_breakdown = self.composite(output, target, auxiliary, epoch)
+        return self.last_breakdown
+
+    def _target_to_onehot(self, target: Tensor, num_classes: int) -> Tensor:
+        indices = target[:, 0].long()
+        onehot = torch.nn.functional.one_hot(indices, num_classes=num_classes)
+        return onehot.permute(0, 4, 1, 2, 3).float()
+
+    def forward(self, output, target, auxiliary=None, epoch=None) -> Tensor:
+        return self.breakdown(output, target, auxiliary, epoch).total
+
+
+def build_composite_loss(
+    config: EnhancementConfig,
+    segmentation_loss: nn.Module,
+    label_manager: Any,
+    projection_loss: Optional[nn.Module] = None,
+    dtm_loss: Optional[nn.Module] = None,
+) -> TrainerLossAdapter:
+    if config.projection_consistency_weight and projection_loss is None:
+        projection_loss = ProjectionConsistencyLoss(config.projection_angles_deg, align_corners=False)
+    if config.dtm_mode != "disabled" and dtm_loss is None:
+        dtm_loss = GenSurfLoss(class_weights=None, ignored_label=-1)
+    return TrainerLossAdapter(
+        CompositeSegmentationLoss(
+            segmentation_loss,
+            config,
+            label_manager,
+            projection_loss=projection_loss,
+            dtm_loss=dtm_loss,
+        )
+    )
+
+
+def build_legacy_loss(preset_name: str, segmentation_loss: nn.Module, label_manager: Any) -> TrainerLossAdapter:
+    return build_composite_loss(legacy_config(preset_name), segmentation_loss, label_manager)
+
+
+def _normalize_legacy_mask(dtm: TensorTree, valid_mask: TensorTree) -> TensorTree:
+    if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
+        if dtm.ndim == 6 and valid_mask.ndim == 5:
+            expanded = valid_mask.unsqueeze(-1)
+            return expanded.expand(dtm.shape)
+        return valid_mask
+    if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
+        if len(dtm) != len(valid_mask):
+            raise ValueError(
+                f"DTM and valid_mask list lengths must match: {len(dtm)} != {len(valid_mask)}"
+            )
+        values = [_normalize_legacy_mask(d, m) for d, m in zip(dtm, valid_mask)]
+        return type(valid_mask)(values) if isinstance(valid_mask, tuple) else values
+    return valid_mask
+
+
+def move_auxiliary_batch(batch: dict, device: torch.device | str) -> AuxiliaryTargets:
+    dtm = batch.get("dtm")
+    valid_mask = batch.get("valid_mask")
+    if dtm is not None and valid_mask is not None:
+        valid_mask = _normalize_legacy_mask(dtm, valid_mask)
+    return AuxiliaryTargets(dtm=dtm, valid_mask=valid_mask).to(device)
+
+
+def calculate_loss(loss: TrainerLossAdapter, output, target, auxiliary=None, epoch=None) -> LossBreakdown:
+    return loss.breakdown(output, target, auxiliary, epoch)
+
+
+def breakdown_log_values(breakdown: LossBreakdown) -> dict[str, Tensor]:
+    return {
+        "loss": breakdown.total,
+        "seg_loss": breakdown.segmentation,
+        "proj_loss": breakdown.projection,
+        "gsl_loss": breakdown.dtm,
+    }
diff --git a/verify_modules.py b/verify_modules.py
index b8e2e72..e105ebc 100644
--- a/verify_modules.py
+++ b/verify_modules.py
@@ -27,12 +27,16 @@ print("-" * 40)
 from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
 from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
 from nnunetv2.network_modules.projection_consistency import (
     ProjectionConsistencyLoss, CombinedSegProjLoss,
 )
 from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
+from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
+from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
+from nnunetv2.training.pipeline_config import EnhancementConfig
+from nnunetv2.training.pipeline_trainer import build_composite_loss
 
 # Build a minimal backbone for testing
 from dynamic_network_architectures.architectures.unet import PlainConvUNet
 backbone = PlainConvUNet(
     input_channels=C, n_stages=4,
     features_per_stage=(8, 16, 32, 64),
@@ -217,13 +221,15 @@ target = torch.randint(0, NUM_CLASSES, (B, 1, 32, 64, 64), device=DEVICE)
 
 l_total = combined(output, target)
 l_seg = combined.seg_loss(output, target)
 
 # Compute proj component separately
 with torch.no_grad():
-    target_oh = combined._target_to_onehot(target, NUM_CLASSES)
+    target_oh = torch.nn.functional.one_hot(
+        target[:, 0].long(), NUM_CLASSES
+    ).permute(0, 4, 1, 2, 3).float()
     l_proj = combined.proj_loss(output, target_oh)
     l_proj_weighted = combined.proj_weight * l_proj
 
 print(f"  Total loss:          {l_total.item():.6f}")
 print(f"  DC+CE loss:          {l_seg.item():.6f}")
 print(f"  ProjCons loss:       {l_proj.item():.6f}")
@@ -319,12 +325,85 @@ loss_change = losses[0] - losses[-1]
 print(f"  Loss sequence: {[f'{l:.4f}' for l in losses]}")
 print(f"  {'[PASS]' if abs(loss_change) > 1e-6 else '[FAIL]'} Loss changes during training "
       f"(delta={loss_change:.6f})")
 
 p7_ok = abs(loss_change) > 1e-6
 
+# ── Proof 8: Unified all-enabled composition path ─────────────────────
+print("\n[Proof 8] Unified all-enabled pipeline + composite loss")
+print("-" * 40)
+
+class TinyVerificationBackbone(nn.Module):
+    def __init__(self):
+        super().__init__()
+        self.decoder = nn.Module()
+        self.projection = nn.Conv3d(C, NUM_CLASSES, kernel_size=1)
+
+    def forward(self, inputs):
+        return self.projection(inputs)
+
+
+class TinyVerificationSegmentationLoss(nn.Module):
+    def forward(self, output, target):
+        return nn.functional.cross_entropy(output, target[:, 0].long())
+
+
+unified_config = EnhancementConfig(
+    enable_anisotropic_attention=True,
+    enable_tubular_enhancement=True,
+    projection_angles_deg=(0.0,),
+    projection_consistency_weight=0.25,
+    dtm_mode="active",
+    dtm_weight=0.5,
+    dtm_activate_after_epoch=0,
+    dtm_active_weight=1.0,
+)
+unified_model = EnhancementPipeline(
+    TinyVerificationBackbone(), C, NUM_CLASSES, unified_config
+).to(DEVICE)
+
+class _UnifiedLabelManager:
+    has_regions = False
+    has_ignore_label = False
+    num_segmentation_heads = NUM_CLASSES
+
+
+unified_loss = build_composite_loss(
+    unified_config,
+    TinyVerificationSegmentationLoss(),
+    _UnifiedLabelManager(),
+).to(DEVICE)
+unified_input = torch.randn(1, C, 8, 8, 8, device=DEVICE)
+unified_target = torch.randint(0, NUM_CLASSES, (1, 1, 8, 8, 8), device=DEVICE)
+unified_dtm = torch.rand(1, 1, 8, 8, 8, NUM_CLASSES, device=DEVICE)
+unified_auxiliary = AuxiliaryTargets(
+    dtm=unified_dtm,
+    valid_mask=torch.ones_like(unified_dtm),
+)
+unified_output = unified_model(unified_input)
+unified_breakdown = unified_loss.breakdown(
+    unified_output, unified_target, unified_auxiliary, epoch=0
+)
+unified_breakdown.total.backward()
+unified_finite = all(
+    torch.isfinite(component).item()
+    for component in (
+        unified_breakdown.total,
+        unified_breakdown.segmentation,
+        unified_breakdown.projection,
+        unified_breakdown.dtm,
+    )
+)
+unified_grad = any(
+    parameter.grad is not None and torch.isfinite(parameter.grad).all()
+    for parameter in unified_model.parameters()
+    if parameter.requires_grad
+)
+p8_ok = unified_finite and unified_grad
+print(f"  {'[PASS]' if p8_ok else '[FAIL]'} all components finite and gradients present")
+
 # ── Summary ──────────────────────────────────────────────────────────
 print("\n" + "=" * 60)
 print("SUMMARY")
 print("=" * 60)
 results = {
     "1. Gradient flow (AnisoAttn)": p1_aniso,
@@ -333,21 +412,22 @@ results = {
     "2. AnisoAttn non-identity": p2_ok,
     "3. TubEnh non-identity": p3_ok,
     "4. ProjCons effectiveness": p4_ok,
     "5. Combined loss balance": p5_ok,
     "6. Projection physics": p6_ok,
     "7. End-to-end training": p7_ok,
+    "8. Unified all-enabled composition": p8_ok,
 }
 
 all_pass = True
 for name, result in results.items():
     status = "[PASS]" if result else "[FAIL]"
     if not result:
         all_pass = False
     print(f"  {status}  Proof {name}")
 
 print("=" * 60)
 if all_pass:
-    print("ALL PROOFS PASSED - All 3 modules are empirically active.")
+    print("ALL PROOFS PASSED - Unified enhancement pipeline is empirically active.")
 else:
     print("SOME PROOFS FAILED - see above for details.")
 print("=" * 60)

