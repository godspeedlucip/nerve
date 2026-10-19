import inspect
import os
from time import time
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

from nnunetv2.training.dataloading.data_loader_dtm import nnUNetDataLoaderDTM
from nnunetv2.training.dataloading.nnunet_dataset_dtm import nnUNetDatasetBlosc2DTM
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.loss.dtm_losses import GenSurfLoss, CombinedLoss, LinearSchedule
from nnunetv2.training.loss.dtm_deep_supervision import DeepSupervisionWrapperDTM
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.pipeline_trainer import (
    LegacyTrainerAssemblyMixin,
    build_composite_loss,
    build_network,
    move_auxiliary_batch,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerDTM(LegacyTrainerAssemblyMixin, nnUNetTrainer):
    """DTM-aware trainer: tracks GSL loss but only uses DC_CE for backprop (monitoring mode)."""

    enhancement_preset = "nnUNetTrainerDTM"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.gsl_alpha = 0.5
        self.gsl_activate_after_epoch = 250

    def _get_deep_supervision_weights(self):
        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()
        return weights

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss_dc_ce = DC_and_BCE_loss({},
                                         {'batch_dice': self.configuration_manager.batch_dice,
                                          'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp,
                                          'return_as_1_minus_dc': True},
                                         use_ignore_label=self.label_manager.ignore_label is not None,
                                         dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss_dc_ce = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                         'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
                                         'return_as_1_minus_dc': True},
                                        {}, weight_ce=1, weight_dice=1,
                                        ignore_label=self.label_manager.ignore_label,
                                        dice_class=MemoryEfficientSoftDiceLoss)
        loss_gsl_raw = GenSurfLoss(class_weights=None, ignored_label=-1)
        linear_schedule = LinearSchedule(num_epochs=1000)

        if self._do_i_compile():
            loss_dc_ce.dc = torch.compile(loss_dc_ce.dc)

        return loss_dc_ce, loss_gsl_raw, linear_schedule

    def _build_gsl_combined_loss(self, loss_dc_ce, loss_gsl, linear_schedule):
        combined = CombinedLoss(dc_ce_loss=loss_dc_ce, gsl_loss=loss_gsl, schedule=linear_schedule)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            combined = DeepSupervisionWrapperDTM(combined, weights)
        return combined

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = build_network(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
                self.enhancement_config(),
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            loss_dc_ce, loss_gsl_raw, linear_schedule = self._build_loss()

            if self.enable_deep_supervision:
                weights = self._get_deep_supervision_weights()
                loss_dc_ce = DeepSupervisionWrapper(loss_dc_ce, weights)
                self.loss_temp = DeepSupervisionWrapper(loss_gsl_raw, weights)
            else:
                self.loss_temp = loss_gsl_raw
            self.loss = build_composite_loss(
                self.enhancement_config(),
                loss_dc_ce,
                self.label_manager,
                dtm_loss=loss_gsl_raw,
            )
            self.linear_schedule = linear_schedule

            self.dataset_class = nnUNetDatasetBlosc2DTM

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = nnUNetDatasetBlosc2DTM

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes,
         ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderDTM(dataset_tr, self.batch_size,
                                    initial_patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderDTM(dataset_val, self.batch_size,
                                     self.configuration_manager.patch_size,
                                     self.configuration_manager.patch_size,
                                     self.label_manager,
                                     oversample_foreground_percent=self.oversample_foreground_percent,
                                     sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                     probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def _move_dtm_to_device(self, dtm, valid_mask):
        auxiliary = move_auxiliary_batch(
            {"dtm": dtm, "valid_mask": valid_mask}, self.device
        )
        return auxiliary.dtm, auxiliary.valid_mask

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

        auxiliary = move_auxiliary_batch(
            {"dtm": dtm, "valid_mask": valid_mask}, self.device
        )
        dtm, valid_mask = auxiliary.dtm, auxiliary.valid_mask

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            breakdown = self.loss.breakdown(output, target, auxiliary, self.current_epoch)
            l = breakdown.total
            l_gsl = breakdown.dtm

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy(), 'gsl_loss': l_gsl.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        dtm = batch.get('dtm')
        valid_mask = batch.get('valid_mask')

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        auxiliary = move_auxiliary_batch(
            {"dtm": dtm, "valid_mask": valid_mask}, self.device
        )
        dtm, valid_mask = auxiliary.dtm, auxiliary.valid_mask

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            breakdown = self.loss.breakdown(output, target, auxiliary, self.current_epoch)
            l = breakdown.total
            l_gsl = breakdown.dtm

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:].sum(0, keepdims=True)
            fp_hard = fp_hard[1:].sum(0, keepdims=True)
            fn_hard = fn_hard[1:].sum(0, keepdims=True)

        return {'loss': l.detach().cpu().numpy(), 'gsl_loss': l_gsl.detach().cpu().numpy(),
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            if 'gsl_loss' in outputs:
                gsl_losses_tr = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gsl_losses_tr, outputs['gsl_loss'])
                gsl_here = np.vstack(gsl_losses_tr).mean()
            else:
                gsl_here = 0.0
        else:
            loss_here = np.mean(outputs['loss'])
            gsl_here = np.mean(outputs.get('gsl_loss', [0.0]))

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('gsl_losses_train', gsl_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        if 'gsl_losses_train' in self.logger.my_fantastic_logging:
            self.print_to_log_file('gsl_loss', np.round(self.logger.my_fantastic_logging['gsl_losses_train'][-1], decimals=4))

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


class _LegacyDTM_GSLTrainer(nnUNetTrainerDTM):
    """DTM trainer where GSL actually participates in backpropagation via CombinedLoss."""

    enhancement_preset = "nnUNetTrainerDTM_GSL"

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            loss_dc_ce, loss_gsl_raw, linear_schedule = self._build_loss()
            # Use CombinedLoss as the primary loss (GSL participates in backprop)
            self.loss = self._build_gsl_combined_loss(loss_dc_ce, loss_gsl_raw, linear_schedule)
            # Keep raw GSL for monitoring (wrapped if deep supervision)
            if self.enable_deep_supervision:
                weights = self._get_deep_supervision_weights()
                self.loss_temp = DeepSupervisionWrapper(loss_gsl_raw, weights)
            else:
                self.loss_temp = loss_gsl_raw
            self.linear_schedule = linear_schedule

            self.dataset_class = nnUNetDatasetBlosc2DTM

            self.was_initialized = True

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

        dtm, valid_mask = self._move_dtm_to_device(dtm, valid_mask)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            if dtm is not None and valid_mask is not None:
                # use epoch 0 for DTM wrappers that don't need it internally
                final_loss, dc_ce_loss, gsl_loss, alpha = self.loss(output, target, dtm, valid_mask, self.current_epoch)
            else:
                # fallback if no DTM available
                if isinstance(self.loss, DeepSupervisionWrapperDTM):
                    # Can't handle without DTM, use loss_temp
                    final_loss = self.loss_temp(output, target)
                    dc_ce_loss = final_loss
                    gsl_loss = torch.tensor(0.0)
                    alpha = 0.0
                else:
                    final_loss = self.loss(output, target)
                    dc_ce_loss = final_loss
                    gsl_loss = torch.tensor(0.0)
                    alpha = 0.0

            # Store for logging
            self._last_dc_ce = dc_ce_loss.detach().cpu().numpy() if isinstance(dc_ce_loss, torch.Tensor) else dc_ce_loss
            self._last_gsl = gsl_loss.detach().cpu().numpy() if isinstance(gsl_loss, torch.Tensor) else gsl_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(final_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': final_loss.detach().cpu().numpy(),
                'gsl_loss': gsl_loss.detach().cpu().numpy() if isinstance(gsl_loss, torch.Tensor) else gsl_loss}


# The public name is intentionally rebound to the same shared implementation
# as the monitor trainer. The preset controls whether DTM contributes to total.
class nnUNetTrainerDTM_GSL(nnUNetTrainerDTM):
    """DTM trainer where GSL participates through the shared composite loss."""

    enhancement_preset = "nnUNetTrainerDTM_GSL"
