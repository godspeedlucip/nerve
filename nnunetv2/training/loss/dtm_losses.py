import torch
import torch.nn as nn
from torch.nn.functional import softmax


class GenSurfLoss(nn.Module):
    """Generalized Surface Loss using distance transform maps.

    Simply computes mean(dtm * pred) for the foreground class.
    """
    def __init__(self, class_weights=None, ignored_label=-1):
        super(GenSurfLoss, self).__init__()
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None
        self.ignored_label = ignored_label
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, valid_mask):
        """
        y_true: (B, 1, D, H, W)
        y_pred: (B, 2, D, H, W)
        valid_mask: (B, 1, D, H, W)
        dtm: (B, channel, D, H, W, class)
        """
        # Reshape the stored DTM from (B, Channel, D, H, W, Class) to
        # (B, Class, D, H, W). AuxiliaryTargets may also provide the already
        # selected foreground map as (B, 1, D, H, W).
        if dtm.ndim == 6:
            dtm = torch.transpose(dtm, dim0=5, dim1=1)
            dtm = torch.squeeze(dtm, dim=5)
        elif dtm.ndim != 5:
            raise ValueError(f"DTM must have 5 or 6 dimensions, got {tuple(dtm.shape)}")

        y_pred = softmax(y_pred, dim=1)
        # Return mean of dtm * pred for the foreground class. Keep the
        # historical unmasked mean; valid_mask remains part of the signature
        # for compatibility with existing DTM trainers.
        dtm_foreground = dtm[:, 1, ...] if dtm.shape[1] > 1 else dtm[:, 0, ...]
        num = torch.mean(dtm_foreground * y_pred[:, 1, ...])
        return torch.mean(num)


class LinearSchedule:
    def __init__(self, num_epochs, init_pause=0):
        if num_epochs <= init_pause:
            raise ValueError("The number of epochs must be greater than the initial pause.")
        self.num_epochs = num_epochs - 1
        self.init_pause = init_pause

    def __call__(self, epoch):
        if epoch > self.num_epochs:
            raise ValueError("The current epoch is greater than the total number of epochs.")
        if epoch > self.init_pause:
            return min(1, max(0, 1.0 - (float(epoch - self.init_pause) / (self.num_epochs - self.init_pause))))
        else:
            return 1.0


class CombinedLoss(nn.Module):
    """
    Loss = DC_CE_loss + alpha * GSL_loss
    alpha is controlled by epoch (fixed to 0.5 after epoch 250 by default).
    """
    def __init__(self, dc_ce_loss, gsl_loss, schedule: LinearSchedule):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.gsl = gsl_loss
        self.schedule = schedule
        self.alpha = 0

    def forward(self, net_output, target, dtm, valid_mask, epoch):
        if epoch >= 250:
            self.alpha = 0.5
        boundary_loss = self.gsl(target, net_output, dtm, valid_mask)
        dc_ce_loss = self.dc_ce(net_output, target)
        final_loss = dc_ce_loss + self.alpha * boundary_loss
        return final_loss, dc_ce_loss, boundary_loss, self.alpha
