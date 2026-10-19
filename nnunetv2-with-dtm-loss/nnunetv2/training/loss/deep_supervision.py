# ## dice_ce_loss和gsl_loss结合在一起的版本
# from torch import nn
# class DeepSupervisionWrapper(nn.Module):
#     def __init__(self, loss, weight_factors=None):
#         """
#         Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
#         inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
#         applied to each entry like this:
#         l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
#         If weights are None, all w will be 1.
#         """
#         super(DeepSupervisionWrapper, self).__init__()
#         assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
#         self.weight_factors = tuple(weight_factors)
#         self.loss = loss
#         print('ok man! DeepSupervisionWrapper is turned on !')

#     # def forward(self, *args):
#     #     assert all([isinstance(i, (tuple, list)) for i in args]), \
#     #         f"all args must be either tuple or list, got {[type(i) for i in args]}"
        
#     #     # assert all([isinstance(i, (tuple, list, int)) for i in args]), \
#     #     #     f"all args must be either tuple or list, got {[type(i) for i in args]}" #主要是输入的epoch是一个int

#     #     # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
#     #     # this code is executed a lot of times!

#     #     if self.weight_factors is None:
#     #         weights = (1, ) * len(args[0])
#     #     else:
#     #         weights = self.weight_factors #默认所有层的权重都是1

#     #     return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
    
#     def forward(self, *args):
#         assert all([isinstance(i, (tuple, list)) for i in args]), \
#             f"all args must be either tuple or list, got {[type(i) for i in args]}"
        
#         # assert all([isinstance(i, (tuple, list, int)) for i in args]), \
#         #     f"all args must be either tuple or list, got {[type(i) for i in args]}" #主要是输入的epoch是一个int

#         # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
#         # this code is executed a lot of times!

#         if self.weight_factors is None:
#             weights = (1, ) * len(args[0])
#         else:
#             weights = self.weight_factors #默认所有层的权重都是1

#         sum_total_loss = 0
#         sum_dice_ce_loss = 0
#         sum_gsl_loss = 0

#         for i, inputs in enumerate(zip(*args)):
#             if (weights[i] != 0.0 ):
#                 final_loss, dice_ce_loss, gsl_loss, alpha = self.loss(*inputs)
#                 # deep_supervision_res.append({'final_loss':final_loss, 'dice_ce_loss':dice_ce_loss, 'gsl_loss':gsl_loss, 'alpha':alpha})
#                 sum_total_loss += weights[i] * final_loss
#                 sum_dice_ce_loss += weights[i] * dice_ce_loss
#                 sum_gsl_loss += weights[i] * gsl_loss
                
#             # return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
#         # return sum_weight_res, deep_supervision_res
#         return sum_total_loss, sum_dice_ce_loss, sum_gsl_loss, alpha

#         # return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])









### dice_ce_loss和gsl_loss区分开的版本

from torch import nn
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        print('ok man! DeepSupervisionWrapper is turned on !')
    
    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        
        # assert all([isinstance(i, (tuple, list, int)) for i in args]), \
        #     f"all args must be either tuple or list, got {[type(i) for i in args]}" #主要是输入的epoch是一个int

        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors #默认所有层的权重都是1
        
        ## 相对应的，loss的forward部分都要返回其最终的loss值
        for i, inputs in enumerate(zip(*args)):
            if (weights[i] != 0.0 ):
                return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        return 0
