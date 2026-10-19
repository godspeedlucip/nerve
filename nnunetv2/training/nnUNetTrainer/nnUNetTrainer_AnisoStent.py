from typing import List, Union

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.network_modules.projection_consistency import (
    ProjectionConsistencyLoss, CombinedSegProjLoss,
)
from nnunetv2.training.pipeline_trainer import (
    LegacyTrainerAssemblyMixin,
    build_composite_loss,
    build_network,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerAnisoStent(LegacyTrainerAssemblyMixin, nnUNetTrainer):
    """
    Trainer that wraps the standard nnU-Net backbone with:

    - **AnisotropicAttention**: suppresses directional streak artifacts from
      limited-angle back-projection.
    - **TubularEnhancement**: recovers thin tubular structures via multi-scale
      dilated convolutions.

    Usage:
        nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerAnisoStent
    """

    enhancement_preset = "nnUNetTrainerAnisoStent"

    @classmethod
    def build_network_architecture(cls, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], tuple],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return build_network(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            cls.enhancement_config(),
        )

    def _build_loss(self):
        return build_composite_loss(self.enhancement_config(), super()._build_loss(), self.label_manager)


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

    enhancement_preset = "nnUNetTrainerAnisoStent_ProjCons"

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.projection_angles_deg = [0, 30, 60]
        self.proj_cons_weight = 0.1
        self.logger.my_fantastic_logging['proj_losses_train'] = list()

    def _build_seg_loss(self):
        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        return loss

    def _build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        seg_loss = self._build_seg_loss()
        if self.enable_deep_supervision:
            seg_loss = DeepSupervisionWrapper(seg_loss, self._get_deep_supervision_weights())
        return build_composite_loss(
            self.enhancement_config(),
            seg_loss,
            self.label_manager,
            projection_loss=self._build_proj_cons_loss(),
        )

    def _get_deep_supervision_weights(self):
        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()
        return weights

    def _build_proj_cons_loss(self):
        return ProjectionConsistencyLoss(
            angles_deg=self.projection_angles_deg,
            align_corners=False,
        )

    def initialize(self):
        if not self.was_initialized:
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            from torch.nn.parallel import DistributedDataParallel as DDP
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            seg_loss = self._build_seg_loss()
            if self.enable_deep_supervision:
                weights = self._get_deep_supervision_weights()
                seg_loss = DeepSupervisionWrapper(seg_loss, weights)

            proj_loss = self._build_proj_cons_loss()

            self.loss = build_composite_loss(
                self.enhancement_config(),
                seg_loss,
                self.label_manager,
                projection_loss=proj_loss,
            )

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("Trainer was already initialized.")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            breakdown = self.loss.breakdown(output, target)
            l_total = breakdown.total
            l_seg = breakdown.segmentation
            l_proj = breakdown.projection

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': l_total.detach().cpu().numpy(),
            'seg_loss': l_seg.detach().cpu().numpy(),
            'proj_loss': l_proj.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            breakdown = self.loss.breakdown(output, target)
            l_total = breakdown.total
            l_seg = breakdown.segmentation
            l_proj = breakdown.projection

        # Dice metric computation on final output
        if self.enable_deep_supervision:
            output_metric = output[0]
            target_metric = target[0] if isinstance(target, list) else target
        else:
            output_metric = output
            target_metric = target

        axes = [0] + list(range(2, output_metric.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_metric) > 0.5).long()
        else:
            output_seg = output_metric.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_metric.shape, device=output_metric.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_metric != self.label_manager.ignore_label).float()
                target_metric_clone = target_metric.clone()
                target_metric_clone[target_metric_clone == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target_metric[:, -1:]
                target_metric_clone = target_metric[:, :-1]
        else:
            mask = None
            target_metric_clone = target_metric

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_metric_clone, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:].sum(0, keepdims=True)
            fp_hard = fp_hard[1:].sum(0, keepdims=True)
            fn_hard = fn_hard[1:].sum(0, keepdims=True)

        return {
            'loss': l_total.detach().cpu().numpy(),
            'seg_loss': l_seg.detach().cpu().numpy(),
            'proj_loss': l_proj.detach().cpu().numpy(),
            'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            from torch import distributed as dist
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            if 'proj_loss' in outputs:
                proj_losses = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(proj_losses, outputs['proj_loss'])
                proj_here = np.vstack(proj_losses).mean()
            else:
                proj_here = 0.0
        else:
            loss_here = np.mean(outputs['loss'])
            proj_here = np.mean(outputs.get('proj_loss', [0.0]))

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('proj_losses_train', proj_here, self.current_epoch)


# ── Ablation Trainers ──────────────────────────────────────────────────


class nnUNetTrainerAnisoStent_AnisoAttnOnly(nnUNetTrainer):
    """
    Ablation: AnisotropicAttention only (no TubularEnhancement).

    Pipeline: Input → AnisotropicAttention → PlainConvUNet → Output
    """

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention

        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)

        class _AnisoOnlyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.aniso_attn = AnisotropicAttention(num_input_channels)
                self.backbone = backbone

            @property
            def decoder(self):
                return self.backbone.decoder

            def forward(self, x):
                return self.backbone(self.aniso_attn(x))

        return _AnisoOnlyUNet()


class nnUNetTrainerAnisoStent_TubEnhOnly(nnUNetTrainer):
    """
    Ablation: TubularEnhancement only (no AnisotropicAttention).

    Pipeline: Input → PlainConvUNet → TubularEnhancement → Output
    """

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement

        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)

        class _TubEnhOnlyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.tub_enh = TubularEnhancement(num_output_channels)

            @property
            def decoder(self):
                return self.backbone.decoder

            def forward(self, x):
                out = self.backbone(x)
                if isinstance(out, list):
                    out[0] = self.tub_enh(out[0])
                    return out
                return self.tub_enh(out)

        return _TubEnhOnlyUNet()


class nnUNetTrainerAnisoStent_ProjConsOnly(nnUNetTrainer):
    """
    Ablation: ProjectionConsistencyLoss on standard PlainConvUNet.

    No AnisotropicAttention, no TubularEnhancement.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.projection_angles_deg = [0, 30, 60]
        self.proj_cons_weight = 0.1

    def _build_proj_cons_loss(self):
        return ProjectionConsistencyLoss(
            angles_deg=self.projection_angles_deg,
            align_corners=False,
        )

    def _build_loss(self):
        # Build raw segmentation loss (no DS wrapping — we'll wrap below)
        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
            seg_loss = DC_and_BCE_loss({},
                                       {'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                       use_ignore_label=self.label_manager.ignore_label is not None,
                                       dice_class=MemoryEfficientSoftDiceLoss)
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
            seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                      {}, weight_ce=1, weight_dice=1,
                                      ignore_label=self.label_manager.ignore_label,
                                      dice_class=MemoryEfficientSoftDiceLoss)
        if self._do_i_compile():
            seg_loss.dc = torch.compile(seg_loss.dc)

        proj_loss = self._build_proj_cons_loss()

        if self.enable_deep_supervision:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            seg_loss = DeepSupervisionWrapper(seg_loss, weights)

        return CombinedSegProjLoss(
            seg_loss=seg_loss,
            proj_loss=proj_loss,
            proj_weight=self.proj_cons_weight,
            label_manager=self.label_manager,
        )


# ── A/B stubs for dual-training compatibility ──────────────────────────
# These differ only by class name so nnU-Net writes checkpoints to separate
# output folders when used with run_training_dual_containment.py.

class nnUNetTrainerAnisoStent_A(nnUNetTrainerAnisoStent):
    """Dual-training stub A for :class:`nnUNetTrainerAnisoStent`."""
    pass


class nnUNetTrainerAnisoStent_B(nnUNetTrainerAnisoStent):
    """Dual-training stub B for :class:`nnUNetTrainerAnisoStent`."""
    pass


class nnUNetTrainerAnisoStent_ProjCons_A(nnUNetTrainerAnisoStent_ProjCons):
    """Dual-training stub A for :class:`nnUNetTrainerAnisoStent_ProjCons`."""
    pass


class nnUNetTrainerAnisoStent_ProjCons_B(nnUNetTrainerAnisoStent_ProjCons):
    """Dual-training stub B for :class:`nnUNetTrainerAnisoStent_ProjCons`."""
    pass
