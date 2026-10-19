
import numpy as np
import ants

def get_dtm(mask, label_list, voxel_spacing):
    # Convert to numpy and get one hot encoding
    # mask_npy = mask.numpy()
    dims = mask.shape
    # target_sapcing = mask.spacing
    mask_onehot = np.zeros((*dims, len(label_list)),dtype=np.float32) # (X, Y, Z, C)
    dtm = np.zeros((*dims, len(label_list)), dtype=np.float32)
    mask_ants = ants.from_numpy(
                    mask,
                    spacing=voxel_spacing
                    # origin=[0.0, 0.0, 0.0],  # 默认原点是 [0.0, 0.0, 0.0]
                    # direction=np.eye(3)    # 默认方向是单位矩阵（轴对齐）
                )
    # for j in range(len(label_list)):
    for j in range(len(label_list)):
        mask_onehot[..., j] = (mask == label_list[j]).astype('float32') #对每个类别j，都创建一个mask图像，相当于手动实现one-hot编码
        # Only compute DTM if class exists in image
        if np.sum(mask_onehot[..., j]) > 0:
            dtm_j = mask_onehot[..., j].reshape(dims)
            dtm_j = mask_ants.new_image_like(data=dtm_j) # 创建一个和mask一样的ants图像，只不过数据是dtm_j
            dtm_j = ants.iMath(dtm_j, 'MaurerDistance') #自动创建距离图
            dtm[..., j] = dtm_j.numpy() #转换为numpy数组
        else: #如果没有元素，就把距离设置为对角线的距离，也就是最大值
            # Otherwise, use diagonal distance of image as the value
            width = dims[0] * voxel_spacing[0]
            height = dims[1] * voxel_spacing[1]
            depth = dims[2] * voxel_spacing[2]
            dtm[..., j] += np.sqrt(width**2 + height**2 + depth**2)
    return dtm



def calculate_class_percentage(mask_array):
    """
    计算任意维度 NumPy 掩码数组中各类的百分占比。

    参数:
        mask_array (np.ndarray): 任意维度的掩码数组（二值或多类）。

    返回:
        dict: 包含类别值和对应百分比的字典。
    """
    # 1. 计算总像素/体素数量
    total_elements = mask_array.size

    # 2. 找到所有唯一的类别值及其计数
    # np.unique 会先将数组展平 (flatten)
    unique_values, counts = np.unique(mask_array, return_counts=True)

    # 3. 计算每个类别的百分比
    percentages = (counts / total_elements) * 100

    # 4. 将结果整理成字典，方便查看
    result_dict = {
        value: percentage
        for value, percentage in zip(unique_values, percentages)
    }

    return result_dict




## 调度器，决定什么时候加入边缘损失
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
        """
        tau_rate: 收敛速率阈值
        tau_dsc: 绝对性能阈值 (DSC_val)
        tau_variance: 优化路径方差阈值
        lag_M: 滞后窗口 M，用于 ΔLoss
        var_window: 计算 mini-batch 方差的窗口大小
        T_min: 最小训练周期，避免过早触发
        """
        self.tau_rate = tau_rate
        self.tau_dsc = tau_dsc
        self.tau_variance = tau_variance
        self.lag_M = lag_M
        self.T_min = T_min

        # 滞后窗口队列（存 Val Dice Loss）
        self.val_dice_loss_history = deque(maxlen=lag_M + 1)

        # mini-batch Dice Loss 队列
        self.train_batch_loss_window = deque(maxlen=var_window)

        self.boundary_activated = False

    def update_train_batch_loss(self, batch_dice_loss):
        """在每个 mini-batch 调用"""
        self.train_batch_loss_window.append(batch_dice_loss)

    def compute_variance(self):
        """计算最近 N 个 mini-batch 的方差"""
        if len(self.train_batch_loss_window) < self.train_batch_loss_window.maxlen:
            return np.inf  # 数据不够，不触发
        return np.var(self.train_batch_loss_window)

    def update_val_metrics(self, val_dice_loss, val_dsc):
        """在每 epoch 验证阶段调用"""
        self.val_dice_loss_history.append(val_dice_loss)
        self.current_val_dsc = val_dsc

    def check_rate(self):
        """标准一：收敛速率 ΔL 判定"""
        if len(self.val_dice_loss_history) < self.val_dice_loss_history.maxlen:
            return False
        L_now = self.val_dice_loss_history[-1]
        L_past = self.val_dice_loss_history[0]
        delta_L = (L_now - L_past) / self.lag_M

        return abs(delta_L) < self.tau_rate

    def check_dsc(self):
        """标准二：绝对性能阈值"""
        return self.current_val_dsc >= self.tau_dsc

    def check_variance(self):
        """标准三：mini-batch 方差稳定"""
        return self.compute_variance() < self.tau_variance

    def should_activate(self, epoch):
        """多因素联合触发协议"""
        if self.boundary_activated:
            return True  # 已激活持续保持

        if epoch < self.T_min:
            return False

        cond1 = self.check_dsc()
        cond2 = self.check_rate()
        cond3 = self.check_variance()

        if cond1 and cond2 and cond3:
            self.boundary_activated = True

        return self.boundary_activated
