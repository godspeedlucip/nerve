import numpy as np
import plyfile,copy
import pandas as pd
import tifffile as tiff
from scipy import io
import os
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk
from scipy.ndimage import convolve, gaussian_filter
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import SimpleITK as sitk # 推荐用于医学图像处理
from skimage.morphology import skeletonize
import numpy as np
from scipy.ndimage import zoom
import os
from os.path import join
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
from scipy.ndimage import uniform_filter
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter


# from gen_stent.utils.base_utils import Normalize, stent_simu, expand_and_smooth_voxel_mask, voxelize_1, voxelize_2, center_point_cloud_to_voxel_space, set_geo, flip_point_cloud, get_min_max, visualize_braid_stent
# from gen_stent.utils.gen_curve import generate_constrained_spiral_curve_iterative, generate_single_sin_curve, generate_3d_sine_curve_pointcloud, generate_random_smooth_curve, create_nurbs_centerline_fixed_length, create_outward_arc_u_shape_point_cloud
# from gen_stent.utils.curve_trans import transform_pointcloud_frenet_serret, transform_pointcloud_parallel, compute_frenet_serret, compute_parallel_transport_frames
# from gen_stent.utils.file_rw import read_ply
# from gen_stent.utils.gen_stent import w_like_Stent, ring_Stent, Stent_Ply, sin_Stent
# from gen_stent.utils.get_stent_para import map_stent_to_curve


# 提取骨架，选择满足距离要求的点
def thicken_by_distance(voxel, r_voxel = 2):
    """
    voxel: 3D numpy array, bool or 0/1, 支架为 True
    r_voxel: 增粗半径（单位：体素）
    """
    flag = True
    skeleton = skeletonize(voxel)

    # 距离到最近“支架体素”的欧氏距离
    dist = distance_transform_edt(~skeleton)

    # 阈值化：距离 <= r 的都纳入支架
    thick_voxel = dist <= r_voxel

    if(len(np.where(voxel > 0)[0])==0):
        flag = False
    # print(len(np.where(voxel > 0)[0]), len(np.where(thick_voxel > 0)[0]))

    voxel[thick_voxel] = 1

    return voxel, skeleton, flag

## 扩张再高斯平滑
def expand_and_smooth_voxel_mask(mask, sigma=1,thred = 0.7,kernel_size = 2,valid_thred=3):
    """
    1. 对体素掩膜进行扩张
    2. 高斯平滑，减轻锯齿
    3. 阈值化回二值
    
    参数:
        mask: np.ndarray, 3D 数组, 值为0或1
        sigma: float, 高斯平滑标准差
    
    返回:
        smoothed_mask: np.ndarray, 3D 二值数组
    """
    # # === Step 1: 邻域扩张 ===
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=int)
    neighbor_sum = convolve(mask, kernel, mode="constant", cval=0)
    updated_mask = mask.copy()
    condition = (mask == 0) & (neighbor_sum >= valid_thred)
    updated_mask[condition] = 1
    
    # === Step 2: 高斯平滑 ===
    smoothed = gaussian_filter(updated_mask.astype(float), sigma=sigma)
    # smoothed = gaussian_filter(mask.astype(float), sigma=sigma)

    # === Step 3: 阈值化 (0.5 可调节) ===
    smoothed_mask = (smoothed >= thred).astype(np.uint8)
    
    return smoothed_mask


def change_radius(d_dist, spacing, stent_point):
    """
    d_dist: 目标中心线的半径 shape: (N, )
    stent_point: 需要调整半径的点云, xy平面以原点为中心, z以0开始
    spacing: voxel space.
    """
    ## 逐个调整半径
    def change_radius_by_z(points: np.ndarray, R_of_z, center_xy=(0.0, 0.0)) -> np.ndarray:
        """
        R_of_z: callable，输入 z (float or array) 输出半径 (same shape)
        """
        cx, cy = center_xy
        x = points[:, 0] - cx
        y = points[:, 1] - cy
        theta = np.arctan2(y, x)

        z = points[:, 2]
        Rz = np.asarray(R_of_z(z), dtype=float)
        if np.any(Rz <= 0):
            raise ValueError("R_of_z produced non-positive radius")

        out = points.copy()
        out[:, 0] = cx + Rz * np.cos(theta)
        out[:, 1] = cy + Rz * np.sin(theta)
        return out

    ## 同一个z轴位置的一起调整
    def change_radius_by_z_binned(
        points,
        R_of_z,
        dz,
        center_xy=(0.0, 0.0)
    ):
        cx, cy = center_xy
        out = points.copy()

        # slab index
        z = points[:, 2]
        z_bin = np.floor(z / dz).astype(int)
        unique_bins = np.unique(z_bin)

        for b in unique_bins:
            mask = (z_bin == b)
            z_center = (b + 0.5) * dz
            Rb = float(R_of_z(z_center))

            x = points[mask, 0] - cx
            y = points[mask, 1] - cy
            theta = np.arctan2(y, x)

            out[mask, 0] = cx + Rb * np.cos(theta)
            out[mask, 1] = cy + Rb * np.sin(theta)
        return out

    target_z_radius = d_dist * spacing[0] #原来是体素空间的半径，将其转换为物理空间，但是半径的数量还是按照体素来的
    stent_point = stent_point * np.array(spacing) #这里需要将其转换为物理空间吗？应该不需要。

    z_min = int(np.min(stent_point[:,2]))
    z_max = int(np.max(stent_point[:,2]) + 1)

    target_z_list = np.linspace(z_min, z_max, len(target_z_radius))

    r_spline = CubicSpline(target_z_list, target_z_radius) #创建一个三维插值器

    ## 其实到底用什么的差别都不是很大
    # modified_stent_points = change_radius_by_z(sin_stent, r_spline)
    modified_stent_points = change_radius_by_z_binned(stent_point, r_spline, dz=0.01)

    return modified_stent_points / np.array(spacing)