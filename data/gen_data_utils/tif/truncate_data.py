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
from skimage.morphology import skeletonize_3d
from skimage.measure import regionprops,label

import json

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def read_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print(f"警告: 文件 {ply_path} 中没有读取到任何点。")
        return pcd, np.array([])
        
    # 将 Open3D 点云对象转换为 NumPy 数组
    point_cloud_np = np.asarray(pcd.points)
    return  point_cloud_np

def read_para(para_path):
    # 读取几何参数
    with open(para_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = loaded_data['x_voxel_size'], loaded_data['y_voxel_size'], loaded_data['z_voxel_size'], loaded_data['x_dim_max'], loaded_data['y_dim_max'], loaded_data['z_dim_max']
    return float(x_voxel_size), float(y_voxel_size), float(z_voxel_size), int(x_dim_max), int(y_dim_max), int(z_dim_max)

def set_geo(DSD  =1250, 
            DSO = 750,
            nDetector = np.array([512,512]),
            dDetector = np.array([0.3, 0.3]), 
            nVoxel = np.array([248, 510, 636]),
            dVoxel = np.array([0.3, 0.3, 0.6]),
            geo_type='cone'):
    # Lets create a geometry object
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = DSD  # Distance Source Detector      (mm)
    geo.DSO = DSO  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = nDetector  # number of pixels              (px)
    geo.dDetector =dDetector  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = nVoxel  # number of voxels              (vx)
    geo.dVoxel= dVoxel # size of each voxel            (mm)
    geo.sVoxel = geo.dVoxel * geo.nVoxel  # total size of the image       (mm)
    
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    # geo.offOrigin = geo.sVoxel /2 

    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # geo.offDetector = geo.sDetector / 2

    geo.accuracy = 0.5  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # geo.mode = "cone"  # Or 'parallel'. Geometry type.
    geo.mode = geo_type  # Or 'parallel'. Geometry type.
    return geo

def truncate_image_single(image, sample_point, para_path, radius=50, more_radius = 5): #截取单个图像
    ## 只截取ROi的区域（因为实际的数据采集中，探测器大小和重建区域大小也只有ROI的大小）
    """
    输入数据为重建得到的图像以及真实的图像，以及采样点。是对单个数据的处理
    """

    _, _, _, x_dim_max, y_dim_max, z_dim_max = read_para(para_path) #存储了体素大小基本信息的参数文件

    ## 得到正方形的roi区域
    # roi_trun = np.zeros([2*(radius + more_radius), 2*(radius + more_radius), 2*(radius + more_radius)]) #包含圆形ROI的外接正方形

    point_cloud_cube = get_filled_bounding_cube_from_sphere(np.array(sample_point), radius + more_radius, resolution_steps=4*(radius+more_radius) )
    roi_image_mask = voxelize(point_cloud_cube, voxel_size = (z_dim_max,x_dim_max,y_dim_max))

    truncate_vol = np.where(roi_image_mask, image, np.min(image))

    ## 将有效区域拿出来
    border = np.array(np.where(roi_image_mask>0)).T
    border_min = np.min(border,axis=0)
    border_max = np.max(border,axis=0)
    start_indices = border_min
    end_indices = border_max + 1
    slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))

    ## 将其值赋值到截断图像上
    roi_trun = truncate_vol[slices]

    return roi_trun


## 生成ROI区域的外接圆（半径应该稍微设置大一点）
def get_filled_bounding_cube_from_sphere(
    center: np.ndarray,
    radius: float,
    resolution_steps: int = 200  # 正方体每个轴上采样的点数 (例如 32x32x32)
) -> np.ndarray:
    """
    根据球体中心和半径，直接生成其外接正方体内部所有点的点云表示。

    Args:
        center: 球体的中心坐标 (Cx, Cy, Cz)。
        radius: 球体的半径 R。
        resolution_steps: 正方体每个轴上均匀采样的点数 (体素分辨率 M)。

    Returns:
        filled_cube_points: 填充了整个正方体体积的点云，形状为 (M^3, 3)。
    """
    
    C_z, C_x, C_y = center
    R = radius

    # --- 1. 确定外接正方体的边界 ---
    
    # 最小和最大边界
    X_min, X_max = C_x - R, C_x + R
    Y_min, Y_max = C_y - R, C_y + R
    Z_min, Z_max = C_z - R, C_z + R
    
    # 检查基本条件
    if R <= 0:
        print("警告：半径必须大于0。返回空点云。")
        return np.empty((0, 3))
    if resolution_steps < 2:
        # 至少需要2个点来定义范围
        resolution_steps = 2
        
    # --- 2. 生成填充整个正方体体积的点云 ---
    
    # 在每个轴上生成均匀分布的坐标
    # 在 [Min, Max] 范围内生成 M 个点
    x_grid = np.linspace(X_min, X_max, resolution_steps)
    y_grid = np.linspace(Y_min, Y_max, resolution_steps)
    z_grid = np.linspace(Z_min, Z_max, resolution_steps)
    
    # 使用 np.meshgrid 生成所有组合的坐标
    # indexing='ij' 适用于图像和体素处理，确保 XYZ 轴正确对应
    Z, X, Y = np.meshgrid(z_grid, x_grid, y_grid, indexing='ij')

    # 将三维网格展平为 (M^3, 3) 的点云形式
    filled_cube_points = np.stack([Z.flatten(), X.flatten(), Y.flatten()], axis=1)
    
    return filled_cube_points


def voxelize(
    point_cloud: np.ndarray, 
    voxel_size = (256, 128, 128), 
    fill_value = 1
) -> np.ndarray:
    """
    使用 NumPy 矢量化操作快速将点云体素化为占用网格。

    Args:
        point_cloud: 形状为 (N, 3) 的点云数组，坐标顺序为 (Z, X, Y)。
        voxel_size: 体素网格的维度 (Depth, Height, Width) 或 (Z, X, Y)。
        fill_value: 用于标记占用体素的值 (通常是 1)。

    Returns:
        voxel: 形状为 voxel_size 的体素占用网格 (NumPy 数组)。
    """
    
    # 1. 坐标取整并转换为整数索引
    indices = np.round(point_cloud).astype(int) 
    
    Z_indices = indices[:, 0]
    X_indices = indices[:, 1]
    Y_indices = indices[:, 2]

    D, H, W = voxel_size # D=Z_max, H=X_max, W=Y_max
    
    # 确保索引在 [0, size-1] 范围内
    valid_mask = (Z_indices >= 0) & (Z_indices < D) & \
                 (X_indices >= 0) & (X_indices < H) & \
                 (Y_indices >= 0) & (Y_indices < W)

    # 3. 筛选有效索引
    valid_Z = Z_indices[valid_mask]
    valid_X = X_indices[valid_mask]
    valid_Y = Y_indices[valid_mask]
    
    # 将三个坐标数组堆叠起来，形成 (M, 3) 形状的有效索引数组
    valid_indices = (valid_Z, valid_X, valid_Y)
    
    # 4. 创建并填充体素网格
    # 初始化体素网格
    voxel = np.zeros(voxel_size, dtype=np.float32) 
    
    # 利用 NumPy 的高级索引（矢量化）一次性设置所有有效体素的值
    # 这是关键步骤，取代了 Python 循环
    voxel[valid_indices] = fill_value
    
    return voxel


## 稀疏采样重建之后直接截取，不然会消耗很多存储空间
def sparse_recon(proj_path, gt_path, tar_range, tar_num, save_dir, para_path, sample_pc_dir, filter = False, radius = 60, more_radius = 4):
    if(filter):
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkYesFilter")
    else:
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkNoFilter")
    
    vol_train_save_path = join(save_path,'vol_train')
    vol_gt_save_path = join(save_path,'vol_gt')
    os.makedirs(vol_train_save_path, exist_ok=True)
    os.makedirs(vol_gt_save_path, exist_ok=True)
    x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = read_para(para_path)
    geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                nDetector=np.array([1024,1024]),
                dDetector=np.array([0.5,0.5]))

    # 180度范围内，尽量多取几例数据来训练
    radius = radius
    more_radius = more_radius

    proj_list = sorted(os.listdir(proj_path))
    gt_list = sorted(os.listdir(gt_path))
    for index,i in enumerate(proj_list):
        if(i.endswith('.npz')):
            proj  = np.load(join(proj_path,proj_list[index]))['data']
            vol_gt = np.load(join(gt_path,gt_list[index]))['data']
        elif(i.endswith('.tif')):
            proj  = tiff.imread(join(proj_path,proj_list[index]))
            vol_gt = tiff.imread(join(gt_path,gt_list[index]))
        else:
            raise ValueError('文件格式应该是npz/tif')
        
        ranges = int(180 / tar_range)
        ranges_to_chose = ranges
        if(ranges >= 10):
            ranges_to_chose = 10 #最多采集10个吧，太多了也不行

        sample_name = i.split('_sample_')[0] # 类似于: mr_001_sample_00.tif
        point_index = int(i.split('_')[-1].split('.')[0])
        sample_points = read_ply(join(sample_pc_dir,sample_name+'.ply')) #
        point = sample_points[point_index]
        vol_gt_truncated = truncate_image_single(vol_gt, point, para_path, radius = radius, more_radius=more_radius)

        for j in random.sample(range(ranges), k=ranges_to_chose):
            # index = np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.uint8)
            index = np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.uint16)
            angles = np.linspace(np.deg2rad(tar_range*j), np.deg2rad(tar_range*(j+1)), tar_num) #120个角度
            proj_part = proj[index]
            
            if(filter):
                imgFDK = algs.fdk(proj_part, geo, angles) #有限角的重建结果
            else:
                imgFDK = algs.fdk_no_filter(proj_part, geo, angles) #fdk不用滤波重建
            
            vol_train_truncated = truncate_image_single(imgFDK, point, para_path, radius = radius, more_radius=more_radius)
            
            file_name = f'{i.split(".")[0]}_{str(int(tar_range*j))}-{str(int(tar_range*(j+1)))}.tif'

            tiff.imwrite(join(vol_train_save_path, file_name), vol_train_truncated)
            tiff.imwrite(join(vol_gt_save_path, file_name), vol_gt_truncated)

            # tiff.imwrite(join(vol_train_save_path, file_name), imgFDK)
            # shutil.copy(join(gt_path,i),join(vol_gt_save_path, file_name))

            # imgFDK、join(gt_path,i)


        print(i,point_index)
        # break

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int)
    parser.add_argument("--tar_num", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--more_radius", type=int)
    args = parser.parse_args()

    tar_range = args.tar_range
    tar_num = args.tar_num 
    radius = args.radius
    more_radius = args.more_radius

    # tar_range = 25
    # tar_num = 4
    # radius = 60
    # more_radius = 4
    para_path = '/data1/liuyang/stent/interior/data/voxel_and_dim_info.json'

    print(f'tar_range: {tar_range}\n',
          f'tar_num: {tar_num}\n',
          f'radius: {radius}\n',
          f'more_radius: {more_radius}\n')

    ## 首先生成训练数据
    train_proj_path = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}/train_data/proj_gt'
    train_gt_path = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}/train_data/vol_gt'
    train_save_dir = f'/data1/liuyang/stent/interior/data/trun_data/radius-{radius}/train_data'
    sample_pc_dir = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}//train_data/point_cloud'
    sparse_recon(train_proj_path,train_gt_path,tar_range,tar_num, train_save_dir, para_path=para_path, sample_pc_dir=sample_pc_dir, filter=False, radius=radius, more_radius=more_radius)

    ## 接着生成测试数据
    test_proj_path = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}/test_data/proj_gt'
    test_gt_path = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}/test_data/vol_gt'
    test_save_dir = f'/data1/liuyang/stent/interior/data/trun_data/radius-{radius}/test_data'
    sample_pc_dir = f'/data1/liuyang/stent/interior/data/src_data/radius-{radius}//test_data/point_cloud'
    sparse_recon(test_proj_path,test_gt_path,tar_range,tar_num, test_save_dir, para_path=para_path, sample_pc_dir=sample_pc_dir, filter=False, radius=radius, more_radius=more_radius)



