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


import SimpleITK as sitk
import os
from tqdm import tqdm
join=os.path.join

def smooth_label_via_sdf(
    label_img: sitk.Image,
    sdf_sigma_mm=0.3,     # 0.2–0.5 mm，越小越安全
    threshold_mm=0.0,
    closing_radius_vox=1 # 可选，0=不做
):
    """
    label_img: 二值label (0/1)
    """
    label = sitk.Cast(label_img > 0, sitk.sitkUInt8)

    # Signed distance field
    sdf = sitk.SignedMaurerDistanceMap(
        label,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True
    )
    sdf = sitk.Cast(sdf, sitk.sitkFloat32)

    # SDF 平滑（单位是 mm）
    if sdf_sigma_mm > 0:
        sdf = sitk.SmoothingRecursiveGaussian(sdf, sdf_sigma_mm)

    # 阈值回 label
    label_smooth = sitk.Cast(sdf > threshold_mm, sitk.sitkUInt8)

    # 可选：closing 修补极小断点
    if closing_radius_vox > 0:
        label_smooth = sitk.BinaryMorphologicalClosing(
            label_smooth, (closing_radius_vox,)*3
        )

    return label_smooth


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--target_spacing",type=float, required=True)
    parser.add_argument("--helper_path",type=str, required=True)
    args = parser.parse_args()
    target_spacing = args.target_spacing
    helper_path = args.helper_path
    
    to_process = ['train', 'test']
    for mode in to_process:
        src_data_dir = join(helper_path, f'{mode}_data/voxel_{target_spacing}/resample')
        save_path = join(helper_path, f'{mode}_data/voxel_{target_spacing}/smooth_resample/')
        os.makedirs(save_path, exist_ok=True)

        for i in tqdm(os.listdir(src_data_dir)):
            nib_mask = sitk.ReadImage(join(src_data_dir, i))
            smooth_mask = smooth_label_via_sdf(nib_mask)
            sitk.WriteImage(smooth_mask, join(save_path, i))



    