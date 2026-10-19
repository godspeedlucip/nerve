from torch import nn


class DeepSupervisionWrapperDTM(nn.Module):
    """
    Same as DeepSupervisionWrapper but forwards extra args (dtm, valid_mask, epoch)
    to the wrapped loss function. The loss is expected to return
    (final_loss, dc_ce_loss, gsl_loss, alpha).
    """
    def __init__(self, loss, weight_factors=None):
        super(DeepSupervisionWrapperDTM, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        """
        args: (net_outputs_tuple, targets_tuple, dtms_tuple, valid_masks_tuple, epoch)
        Each tuple contains multi-scale versions.
        """
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"

        if self.weight_factors is None:
            weights = (1,) * len(args[0])
        else:
            weights = self.weight_factors

        sum_total_loss = 0
        sum_dc_ce_loss = 0
        sum_gsl_loss = 0
        alpha = 0

        for i, inputs in enumerate(zip(*args)):
            if weights[i] != 0.0:
                final_loss, dc_ce_loss, gsl_loss, alpha = self.loss(*inputs)
                sum_total_loss += weights[i] * final_loss
                sum_dc_ce_loss += weights[i] * dc_ce_loss
                sum_gsl_loss += weights[i] * gsl_loss

        return sum_total_loss, sum_dc_ce_loss, sum_gsl_loss, alpha
