COMMITS
9b8f19f (HEAD -> master) refactor: assemble legacy trainers from presets
STAT
 .../task-4-report.md                               |  52 ++++++++
 nnunetv2/tests/unit/test_legacy_presets.py         | 113 ++++++++++++++++
 .../nnUNetTrainer/dual_network_trainers.py         |  25 ++++
 .../training/nnUNetTrainer/nnUNetTrainerDTM.py     |  84 +++++++-----
 .../nnUNetTrainer/nnUNetTrainer_AnisoStent.py      |  97 ++++++--------
 nnunetv2/training/pipeline_trainer.py              | 148 +++++++++++++++++++++
 6 files changed, 430 insertions(+), 89 deletions(-)
DIFF
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
diff --git a/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py b/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
index 30b4016..e617520 100644
--- a/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
+++ b/nnunetv2/training/nnUNetTrainer/dual_network_trainers.py
@@ -1,13 +1,14 @@
 import torch
 
 from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
+from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDTM import nnUNetTrainerDTM, nnUNetTrainerDTM_GSL
 
 
 class nnUNetTrainerA(nnUNetTrainer):
     """
     Trainer class for network A.
     Kept separate so nnU-Net writes A and B checkpoints into different result folders.
     """
     def __init__(
         self,
         plans: dict,
@@ -26,10 +27,34 @@ class nnUNetTrainerB(nnUNetTrainer):
     """
     def __init__(
         self,
         plans: dict,
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
@@ -14,28 +14,36 @@ from torch.nn.parallel import DistributedDataParallel as DDP
 from torch import distributed as dist
 
 from nnunetv2.training.dataloading.data_loader_dtm import nnUNetDataLoaderDTM
 from nnunetv2.training.dataloading.nnunet_dataset_dtm import nnUNetDatasetBlosc2DTM
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
 
     def _get_deep_supervision_weights(self):
         deep_supervision_scales = self._get_deep_supervision_scales()
         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
         if self.is_ddp and not self._do_i_compile():
@@ -82,48 +90,53 @@ class nnUNetTrainerDTM(nnUNetTrainer):
         return combined
 
     def initialize(self):
         if not self.was_initialized:
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
 
             self.optimizer, self.lr_scheduler = self.configure_optimizers()
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
             raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")
 
     def get_dataloaders(self):
         if self.dataset_class is None:
@@ -178,57 +191,48 @@ class nnUNetTrainerDTM(nnUNetTrainer):
             mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                       transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                       num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                       pin_memory=self.device.type == 'cuda',
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
             self.grad_scaler.update()
         else:
             l.backward()
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
@@ -241,33 +245,31 @@ class nnUNetTrainerDTM(nnUNetTrainer):
         target = batch['target']
         dtm = batch.get('dtm')
         valid_mask = batch.get('valid_mask')
 
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
 
         if self.label_manager.has_regions:
             predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
         else:
@@ -339,23 +341,25 @@ class nnUNetTrainerDTM(nnUNetTrainer):
             self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
             self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
             self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
 
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
                                                                    self.dataset_json)
 
             self.network = self.build_network_architecture(
                 self.configuration_manager.network_arch_class_name,
@@ -435,10 +439,18 @@ class nnUNetTrainerDTM_GSL(nnUNetTrainerDTM):
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
             self.grad_scaler.step(self.optimizer)
             self.grad_scaler.update()
         else:
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
@@ -1,75 +1,85 @@
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
 
     Uses ``CombinedSegProjLoss`` so that ``trainer.loss(output, target)`` returns
     the combined DC+CE + ProjCons loss.  This is compatible with both standalone
     training and external loops (e.g. ``run_training_dual_containment.py``).
 
     Configurable attributes (set in subclass or via plans/dataset_json):
         - ``projection_angles_deg``: list of projection angles (default [0, 30, 60])
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
 
     def _build_seg_loss(self):
         if self.label_manager.has_regions:
             from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
@@ -84,20 +94,33 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
             from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                   {}, weight_ce=1, weight_dice=1,
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
             weights[-1] = 0
         weights = weights / weights.sum()
         return weights
 
@@ -137,25 +160,25 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                 self.network = DDP(self.network, device_ids=[self.local_rank])
 
             seg_loss = self._build_seg_loss()
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
             raise RuntimeError("Trainer was already initialized.")
 
     def train_step(self, batch: dict) -> dict:
         data = batch['data']
@@ -163,41 +186,24 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
 
         data = data.to(self.device, non_blocking=True)
         if isinstance(target, list):
             target = [i.to(self.device, non_blocking=True) for i in target]
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
             self.grad_scaler.update()
         else:
             l_total.backward()
             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
@@ -215,39 +221,24 @@ class nnUNetTrainerAnisoStent_ProjCons(nnUNetTrainerAnisoStent):
 
         data = data.to(self.device, non_blocking=True)
         if isinstance(target, list):
             target = [i.to(self.device, non_blocking=True) for i in target]
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
             output_metric = output
             target_metric = target
 
         axes = [0] + list(range(2, output_metric.ndim))
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

