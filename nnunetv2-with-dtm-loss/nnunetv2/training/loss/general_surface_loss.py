import torch
import torch.nn as nn
from torch.nn.functional import softmax, one_hot

## GSL里面混杂了DiceCELoss，所以暂且把这个也加进来


import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn





class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


# ## 同时计算前景+背景的损失
# class GenSurfLoss(nn.Module):
#     def __init__(self, class_weights=None, ignored_label=-1):
#         super(GenSurfLoss, self).__init__()

#         # Define class weight scheme
#         # Move weights to cuda if already given by user
#         if not(class_weights is None):
#             self.class_weights = torch.Tensor(class_weights).to("cuda")
#         else:
#             self.class_weights = None

#         self.ignored_label = ignored_label
#         self.smooth = 1e-6
#         self.axes = (2, 3, 4)

#     def get_one_hot(self, y_true, n_classes):
#         y_true = y_true.to(torch.int64)
#         y_true = one_hot(y_true, num_classes=n_classes) #(B, Channel, D, H, W, Class)
#         y_true = torch.transpose(y_true, dim0=5, dim1=1)
#         y_true = torch.squeeze(y_true, dim=5) #(B, Class, D, H, W)
#         y_true = y_true.to(torch.int8)
#         return y_true

#     def forward(self, y_true, y_pred, dtm, valid_mask):
#         """
#         y_true: (B, 1, D, H, W)
#         y_pred: (B, 2, D, H ,W)
#         valid_mask: (B, 1, D, H, W)
#         dtm: (B, channel=11, D, H, W, class)
#         """

#         # 首先抽取有效区域. 对于mask来说，本来就不该有channel这一说，所以这里直接将其替换为class
#         ### 理论上来说，只要dtm的值为0，就相当于没有参与到损失的计算中来，因此好像valid_mask的作用可有可无。但还是可以先试试

#         ## 这里是把(B, Channel, X, Y, Z) -> (B, Class, X, Y, Z)
#         valid_mask_stack = torch.stack((valid_mask, valid_mask), dim=-1)
#         valid_mask_stack = torch.transpose(valid_mask_stack, dim0=5, dim1=1)
#         valid_mask_stack = torch.squeeze(valid_mask_stack, dim=5) #(B, Class, D, H, W)

#         ## 把dtm从 (B, Channel, D, H, W, Class) -> (B, Class, D, H, W)
#         dtm = torch.transpose(dtm, dim0=5, dim1=1)
#         dtm = torch.squeeze(dtm, dim=5)

#         # dtm = dtm * valid_mask_stack

#         # Prepare inputs
#         y_true = self.get_one_hot(y_true, y_pred.shape[1]) #(B, Class, D, H, W)
#         y_pred = softmax(y_pred, dim=1) # 在dim=1维度上（也就是Class维度上）做归一化，也就是让每个体素的类别概率之和为1

#         if self.class_weights is None:
#             y_true_mask = y_true!=self.ignored_label #只计算有效值的class_weight. 输入的y_true实际上的值是(-1, 0, 1)
#             # valid_mask_weight = torch.stack((valid_mask, valid_mask), dim=1) #(B, class, D, H, W)
#             # print('loss func: ',y_true_mask.shape, valid_mask_weight.shape, y_true.shape)
#             # class_weights = torch.sum(y_true * y_true_mask.float() * valid_mask_weight.float(), dim=self.axes) #(B, Class)


#             # class_weights = torch.sum(y_true * y_true_mask.float() * valid_mask_stack.float(), dim=self.axes) #(B, Class)
#             class_weights = torch.sum(y_true * y_true_mask.float(), dim=self.axes) #(B, Class)
#             class_weights = 1. / (torch.square(class_weights) + 1.)
#         else:
#             class_weights = self.class_weights

#         # print(class_weights)

#         # Compute boundary loss
#         # Flip each one-hot encoded class
#         y_worst = torch.square(1.0 - y_true) #y_worst就是y_true的反转. 这里进行square是为了把-1变为1,因为y_true是二值mask

#         num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=self.axes)
#         num *= class_weights

#         # den = torch.sum(valid_mask_stack * torch.square(dtm * (y_worst - y_true)), axis=self.axes)
#         den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=self.axes)
#         den *= class_weights
#         den += self.smooth

#         boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
#         boundary_loss = torch.mean(boundary_loss)
#         boundary_loss = 1. - boundary_loss
#         # print('boundary_loss: ',boundary_loss)
#         return boundary_loss


# ## 只计算前景的损失的版本
# class GenSurfLoss(nn.Module):
#     def __init__(self, class_weights=None, ignored_label=-1):
#         super(GenSurfLoss, self).__init__()

#         # Define class weight scheme
#         # Move weights to cuda if already given by user
#         if not(class_weights is None):
#             self.class_weights = torch.Tensor(class_weights).to("cuda")
#         else:
#             self.class_weights = None

#         self.ignored_label = ignored_label
#         self.smooth = 1e-6
#         self.axes = (2, 3, 4) #只要前景的损失，那么对应的patch_3d_size就是1、2、3维度的数据

#     def get_one_hot(self, y_true, n_classes):
#         y_true = y_true.to(torch.int64)
#         y_true = one_hot(y_true, num_classes=n_classes) #(B, Channel, D, H, W, Class)
#         y_true = torch.transpose(y_true, dim0=5, dim1=1)
#         y_true = torch.squeeze(y_true, dim=5) #(B, Class, D, H, W)
#         y_true = y_true.to(torch.int8)
#         return y_true

#     def forward(self, y_true, y_pred, dtm, valid_mask):
#         """
#         y_true: (B, 1, D, H, W)
#         y_pred: (B, 2, D, H ,W)
#         valid_mask: (B, 1, D, H, W)
#         dtm: (B, channel=1, D, H, W, class)
#         """
#         # print(dtm.shape, y_pred.shape, y_true.shape)
#         # 首先抽取有效区域. 对于mask来说，本来就不该有channel这一说，所以这里直接将其替换为class
#         ### 理论上来说，只要dtm的值为0，就相当于没有参与到损失的计算中来，因此好像valid_mask的作用可有可无。但还是可以先试试

#         ## 这里是把(B, Channel, X, Y, Z) -> (B, Class, X, Y, Z)
#         valid_mask_stack = torch.stack((valid_mask, valid_mask), dim=-1)
#         valid_mask_stack = torch.transpose(valid_mask_stack, dim0=5, dim1=1)
#         valid_mask_stack = torch.squeeze(valid_mask_stack, dim=5) #(B, Class, D, H, W)

#         ## 把dtm从 (B, Channel, D, H, W, Class) -> (B, Class, D, H, W)
#         dtm = torch.transpose(dtm, dim0=5, dim1=1)
#         dtm = torch.squeeze(dtm, dim=5)
#         dtm = dtm[:,1,...] #只取一个长度的话，其shape就会自动忽略这一项，变为(B, D, H, W)

#         # dtm = dtm * valid_mask_stack

#         # Prepare inputs
#         y_true = self.get_one_hot(y_true, y_pred.shape[1]) #(B, Class, D, H, W)
#         y_pred = softmax(y_pred, dim=1) # 在dim=1维度上（也就是Class维度上）做归一化，也就是让每个体素的类别概率之和为1


#         # y_true = y_true[:,1,...] #取前景的部分
#         y_pred = y_pred[:,1,...] #取前景的部分 #shape ( B, D, H, W )

#         ## 手动插入一个维度
#         y_pred = y_pred[:,None,...]
#         y_true = y_true[:,None,...]
#         dtm = dtm[:,None,...]

#         # Compute boundary loss
#         # Flip each one-hot encoded class
#         y_worst = torch.square(1.0 - y_true) #y_worst就是y_true的反转

#         # print(dtm.shape, y_pred.shape, y_true.shape)
#         num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=self.axes)

#         # den = torch.sum(valid_mask_stack * torch.square(dtm * (y_worst - y_true)), axis=self.axes)
#         den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=self.axes)
#         den += self.smooth

#         boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
#         boundary_loss = torch.mean(boundary_loss)
#         boundary_loss = 1. - boundary_loss

#         return boundary_loss




## 不用general surface loss, 而是直接输出概率图 * dtm求和
class GenSurfLoss(nn.Module):
    def __init__(self, class_weights=None, ignored_label=-1):
        super(GenSurfLoss, self).__init__()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not(class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.ignored_label = ignored_label
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, valid_mask):
        """
        y_true: (B, 1, D, H, W)
        y_pred: (B, 2, D, H ,W)
        valid_mask: (B, 1, D, H, W)
        dtm: (B, channel=11, D, H, W, class)
        """
        ## 这里是把(B, Channel, X, Y, Z) -> (B, Class, X, Y, Z)
        valid_mask_stack = torch.stack((valid_mask, valid_mask), dim=-1)
        valid_mask_stack = torch.transpose(valid_mask_stack, dim0=5, dim1=1)
        valid_mask_stack = torch.squeeze(valid_mask_stack, dim=5) #(B, Class, D, H, W)

        ## 把dtm从 (B, Channel, D, H, W, Class) -> (B, Class, D, H, W)
        dtm = torch.transpose(dtm, dim0=5, dim1=1)
        dtm = torch.squeeze(dtm, dim=5)

        # dtm = dtm * valid_mask_stack

        # Prepare inputs
        y_pred = softmax(y_pred, dim=1) # 在dim=1维度上（也就是Class维度上）做归一化，也就是让每个体素的类别概率之和为1

        # ## 试图将整个损失归一化到0~1之间
        # num = torch.sum(dtm * y_pred, axis=self.axes)
        # den = dtm + self.smooth
        # den = torch.sum(den, axis=self.axes)
        # boundary_loss = (num / den) #其shape应该是(B, C)
        # return torch.mean(boundary_loss[:,1]) #只返回前景元素的损失

        ## 直接返回平均的距离
        num = torch.mean(dtm[:,1,...] * y_pred[:,1,...]) # 理论上应该是负数是最佳
        return torch.mean(num) #只返回前景元素的损失

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
    Loss = α(epoch) * DC_CE  +  (1 - α(epoch)) * GSL
    """
    def __init__(self, dc_ce_loss, gsl_loss, schedule: LinearSchedule):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.gsl = gsl_loss
        self.schedule = schedule
        self.alpha = 0

    def forward(self, net_output, target, dtm, valid_mask, epoch):
        ## 权重固定为0.1
        # self.alpha = 0.1

        ## 权重之和为1
        # self.alpha = self.schedule(epoch)

        if(epoch >= 250): #手动设置
            self.alpha=0.5

        boundary_loss = self.gsl(target, net_output, dtm, valid_mask)
        dc_ce_loss = self.dc_ce(net_output, target)
        # final_loss = self.alpha * dc_ce_loss + (1 - self.alpha) * boundary_loss # 线性调整权重
        final_loss = dc_ce_loss + self.alpha * boundary_loss # 固定权重
        return final_loss, dc_ce_loss, boundary_loss, self.alpha


# ## 可以自动调度的版本
# class CombinedLoss(nn.Module):
#     """
#     Loss = α(epoch) * DC_CE  +  (1 - α(epoch)) * GSL
#     """
#     def __init__(self, dc_ce_loss, gsl_loss, schedule: LinearSchedule):
#         super().__init__()
#         self.dc_ce = dc_ce_loss
#         self.gsl = gsl_loss
#         self.schedule = schedule
#         self.alpha = 0

#     def set_alpha(self, alpha):
#         self.alpha = alpha

#     def forward(self, net_output, target, dtm, valid_mask, epoch):
#         alpha = self.schedule(epoch)
#         # print(alpha)
#         boundary_loss = self.gsl(target, net_output, dtm, valid_mask)
#         dc_ce_loss = self.dc_ce(net_output, target)
#         # final_loss = alpha * dc_ce_loss + (1 - alpha) * boundary_loss
#         final_loss = dc_ce_loss + self.alpha * boundary_loss #修改一下权重看看
#         # print(boundary_loss, dc_ce_loss, self.alpha)
#         # return final_loss, dc_ce_loss, boundary_loss, alpha
#         return final_loss, dc_ce_loss, boundary_loss, self.alpha


