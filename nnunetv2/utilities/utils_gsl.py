import numpy as np


def get_dtm(mask, label_list, voxel_spacing):
    try:
        import ants
    except ImportError:
        raise ImportError("ANTsPy is required for DTM computation. Install with: pip install antspyx")
    dims = mask.shape
    mask_onehot = np.zeros((*dims, len(label_list)), dtype=np.float32)
    dtm = np.zeros((*dims, len(label_list)), dtype=np.float32)
    mask_ants = ants.from_numpy(
        mask,
        spacing=voxel_spacing
    )
    for j in range(len(label_list)):
        mask_onehot[..., j] = (mask == label_list[j]).astype('float32')
        if np.sum(mask_onehot[..., j]) > 0:
            dtm_j = mask_onehot[..., j].reshape(dims)
            dtm_j = mask_ants.new_image_like(data=dtm_j)
            dtm_j = ants.iMath(dtm_j, 'MaurerDistance')
            dtm[..., j] = dtm_j.numpy()
        else:
            width = dims[0] * voxel_spacing[0]
            height = dims[1] * voxel_spacing[1]
            depth = dims[2] * voxel_spacing[2]
            dtm[..., j] += np.sqrt(width**2 + height**2 + depth**2)
    return dtm


def calculate_class_percentage(mask_array):
    total_elements = mask_array.size
    unique_values, counts = np.unique(mask_array, return_counts=True)
    percentages = (counts / total_elements) * 100
    result_dict = {
        value: percentage
        for value, percentage in zip(unique_values, percentages)
    }
    return result_dict


import torch
import numpy as np
from collections import deque


class BoundaryActivationScheduler:
    def __init__(self,
                 tau_rate=1e-4,
                 tau_dsc=0.82,
                 tau_variance=1e-5,
                 lag_M=5,
                 var_window=100,
                 T_min=30):
        self.tau_rate = tau_rate
        self.tau_dsc = tau_dsc
        self.tau_variance = tau_variance
        self.lag_M = lag_M
        self.T_min = T_min

        self.val_dice_loss_history = deque(maxlen=lag_M + 1)
        self.train_batch_loss_window = deque(maxlen=var_window)
        self.boundary_activated = False

    def update_train_batch_loss(self, batch_dice_loss):
        self.train_batch_loss_window.append(batch_dice_loss)

    def compute_variance(self):
        if len(self.train_batch_loss_window) < self.train_batch_loss_window.maxlen:
            return np.inf
        return np.var(self.train_batch_loss_window)

    def update_val_metrics(self, val_dice_loss, val_dsc):
        self.val_dice_loss_history.append(val_dice_loss)
        self.current_val_dsc = val_dsc

    def check_rate(self):
        if len(self.val_dice_loss_history) < self.val_dice_loss_history.maxlen:
            return False
        L_now = self.val_dice_loss_history[-1]
        L_past = self.val_dice_loss_history[0]
        delta_L = (L_now - L_past) / self.lag_M
        return abs(delta_L) < self.tau_rate

    def check_dsc(self):
        return self.current_val_dsc >= self.tau_dsc

    def check_variance(self):
        return self.compute_variance() < self.tau_variance

    def should_activate(self, epoch):
        if self.boundary_activated:
            return True
        if epoch < self.T_min:
            return False
        cond1 = self.check_dsc()
        cond2 = self.check_rate()
        cond3 = self.check_variance()
        if cond1 and cond2 and cond3:
            self.boundary_activated = True
        return self.boundary_activated
