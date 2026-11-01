## 将数据重采样到0.2, 0.2, 0.2的量级

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import plyfile,copy
import pandas as pd
import tifffile as tiff
import torch

from scipy import io
import os
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from scipy.signal import find_peaks
import open3d as o3d
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk

from geomdl import NURBS, utilities
from scipy.ndimage import convolve, gaussian_filter
import tigre
import tigre.algorithms as algs

import open3d as o3d

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib

import random
import tqdm

import SimpleITK as sitk # 推荐用于医学图像处理
from skimage.morphology import skeletonize
from skimage.measure import regionprops,label

import json

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def resample_nifti_to_spacing(img, target_spacing=(1.0, 1.0, 1.0), interpolation_order=1, thred=0.5, need_thred = True):
    """
    使用 NiBabel 和 SciPy 对 NIfTI 图像进行重采样到指定体素大小。

    Args:
        input_path (str): 输入 .nii 或 .nii.gz 文件的路径。
        output_path (str): 输出重采样后文件的路径。
        target_spacing (tuple): 目标体素大小 (X, Y, Z)，单位通常是毫米(mm)。
        interpolation_order (int): SciPy 插值阶数。0=最近邻, 1=线性, 3=三次(默认)。
    """
    data = img.get_fdata()
    affine = img.affine
    
    # 从仿射矩阵中提取原始体素大小
    # 仿射矩阵的 (3x3) 旋转/缩放部分每列的欧几里得范数即为体素大小
    original_spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # 2. 计算缩放因子 (Zoom Factors)
    # 缩放因子 = 原始大小 / 目标大小
    zoom_factors = original_spacing / np.array(target_spacing)

    # 3. 应用插值重采样
    # 使用 SciPy 的 zoom 函数对数据进行插值
    resampled_data = zoom(
        data,
        zoom_factors,
        order=interpolation_order,
        mode='nearest' # 处理边界条件
    )
    if(need_thred):
        resampled_data[resampled_data >= thred]  =1
        resampled_data[resampled_data < thred]  = 0
        resampled_data.astype(np.int8)

    # 4. 计算新的仿射矩阵 (New Affine)
    # 保持图像在物理空间中的位置不变 (即坐标原点不变)，
    # 只修改体素大小部分。通过将仿射矩阵的每一列除以对应的缩放因子来实现。
    new_affine = affine.copy()
    for i in range(3):
        # 将原始仿射矩阵的 (3x3) 缩放/旋转部分按比例调整
        # 缩放因子 (zoom_factors[i]) 越大，新的体素在物理空间中越小，
        # 因此仿射矩阵中的对应值需要减小 (除以 zoom_factors[i])。
        new_affine[:3, i] = affine[:3, i] / zoom_factors[i]
    
    
    # 5. 创建并保存新的 NiBabel 图像
    # 使用重采样后的数据和新的仿射矩阵创建新的 Nifti1Image
    # resampled_img = nib.Nifti1Image(resampled_data.astype(img.get_data_dtype()), new_affine, header=img.header)
    resampled_img = nib.Nifti1Image(resampled_data, new_affine, header=img.header)
    
    # 可选：更新头文件中的体素大小信息 (pixdim)，虽然通常依赖 affine 矩阵
    resampled_img.header.set_zooms(target_spacing) 
    
    return resampled_img


# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
#     parser.add_argument("--target_spacing",type=float, required=True)
#     parser.add_argument("--root_data_path",type=str, required=True)
#     parser.add_argument("--helper_path",type=str, required=True)
#     args = parser.parse_args()
#     target_spacing = args.target_spacing
#     src_data_dir = args.root_data_path
#     helper_path = args.helper_path
    
#     to_process = ['train', 'test']
#     for mode in to_process:
#         src_data_path = join(src_data_dir, f"{mode}_data")
#         save_path = join(helper_path, f'{mode}_data/voxel_{target_spacing}/resample/')
#         os.makedirs(save_path, exist_ok=True)

#         for i in tqdm(os.listdir(src_data_path)):
#             nib_mask = nib.load(join(src_data_path, i))
#             resampled_nib_mask = resample_nifti_to_spacing(nib_mask, (target_spacing, target_spacing, target_spacing), need_thred=True, interpolation_order=1) #默认是线性插值
#             nib.save(resampled_nib_mask, join(save_path, i))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--target_spacing",type=float, required=True)
    parser.add_argument("--root_data_path",type=str, required=True)
    parser.add_argument("--helper_path",type=str, required=True)
    args = parser.parse_args()
    target_spacing = args.target_spacing
    src_data_dir = args.root_data_path
    helper_path = args.helper_path
    
    to_process = ['train', 'test']
    for mode in to_process:
        src_data_path = join(src_data_dir, f"{mode}_data/art_only_label")
        save_path = join(helper_path, f'{mode}_data/voxel_{target_spacing}/resample/')
        os.makedirs(save_path, exist_ok=True)

        for i in tqdm(os.listdir(src_data_path)):
            nib_mask = nib.load(join(src_data_path, i))
            resampled_nib_mask = resample_nifti_to_spacing(nib_mask, (target_spacing, target_spacing, target_spacing), need_thred=True, interpolation_order=1) #默认是线性插值
            nib.save(resampled_nib_mask, join(save_path, i))



    