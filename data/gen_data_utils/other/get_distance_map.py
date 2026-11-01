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


from scipy.ndimage import distance_transform_edt



if __name__=='__main__':
    ## 应该是要对resample之后的数据去提取其距离图
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--target_spacing",type=float, required=True)
    parser.add_argument("--helper_path",type=str, required=True)
    args = parser.parse_args()
    target_spacing = args.target_spacing
    helper_path = args.helper_path
    
    to_process = ['train', 'test']
    # to_process = ['test']
    for mode in to_process:
        src_data_dir = join(helper_path, f'{mode}_data/voxel_{target_spacing}/resample')
        save_path = join(helper_path, f'{mode}_data/voxel_{target_spacing}/distance_map/')

        os.makedirs(save_path, exist_ok=True)

        for i in tqdm(os.listdir(src_data_dir)):
            nib_mask = nib.load(join(src_data_dir, i))
            distance_map = distance_transform_edt(nib_mask.get_fdata()) #默认是线性插值
            np.savez_compressed(join(save_path, i.split('.')[0]+'.npz'), data=distance_map)



