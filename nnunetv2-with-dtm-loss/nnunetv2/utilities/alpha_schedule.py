
import os
import json

import ants
import random
import socket
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from models.get_model import get_model
from metrics.metrics import dice_sitk, hausdorff, surface_hausdorff


class StepSchedule:
    def __init__(self, num_epochs, step_length):
        self.step_length = step_length
        self.num_steps = num_epochs // step_length
        self.num_epochs = num_epochs + step_length

    def __call__(self, epoch):
        if epoch >= self.num_epochs - self.step_length:
            return 0
        step = epoch // self.step_length
        return max(0, 1 - step / self.num_steps)


class CosineSchedule:
    def __init__(self, num_epochs, min_val=0, max_val=1):
        self.num_epochs = num_epochs - 1
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, epoch):
        cos_out = (1 + np.cos(np.pi * epoch / self.num_epochs)) / 2
        return self.min_val + (self.max_val - self.min_val) * cos_out

# 线性计划：在init_pause之前都是保持alpha为1，然后线性下降到0
class LinearSchedule:
    def __init__(self, num_epochs, init_pause):
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


class AlphaSchedule:
    def __init__(self, n_epochs, schedule, **kwargs):
        self.schedule = schedule
        self.linear = LinearSchedule(n_epochs, init_pause=kwargs["init_pause"])
        self.step = StepSchedule(n_epochs - kwargs["step_length"], step_length=kwargs["step_length"])
        self.cosine = CosineSchedule(n_epochs)

    def __call__(self, epoch):
        if self.schedule == "linear":
            return self.linear(epoch)
        elif self.schedule == "step":
            return self.step(epoch)
        elif self.schedule == "cosine":
            return self.cosine(epoch).astype('float32')
        else:
            raise ValueError("Enter valid schedule type")
        



def get_dtm(mask, label_list):
    # Convert to numpy and get one hot encoding
    mask_npy = mask.numpy()
    dims = mask_npy.shape
    target_sapcing = mask.spacing
    mask_onehot = np.zeros((*dims, len(label_list)),dtype=np.float32) # (X, Y, Z, C)
    dtm = np.zeros((*dims, len(label_list)), dtype=np.float32)
    for j in range(len(label_list)):
        mask_onehot[..., j] = (mask_npy == label_list[j]).astype('float32') #对每个类别j，都创建一个mask图像，相当于手动实现one-hot编码

        # Only compute DTM if class exists in image
        if np.sum(mask_onehot[..., j]) > 0:
            dtm_j = mask_onehot[..., j].reshape(dims)
            dtm_j = mask.new_image_like(data=dtm_j) # 创建一个和mask一样的ants图像，只不过数据是dtm_j
            dtm_j = ants.iMath(dtm_j, 'MaurerDistance') #自动创建距离图
            dtm[..., j] = dtm_j.numpy() #转换为numpy数组
        else: #如果没有元素，就把距离设置为对角线的距离，也就是最大值
            # Otherwise, use diagonal distance of image as the value
            print(j, 'yes i do')
            width = dims[0] * target_sapcing[0]
            height = dims[1] * target_sapcing[1]
            depth = dims[2] * target_sapcing[2]
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

