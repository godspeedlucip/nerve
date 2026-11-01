
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

def get_rectangle_roi(roi_center,roi_size): #获得长方形区域的roi坐标
    # box_min = (roi_center - roi_size / 2.0).astype(np.uint8)
    # box_max = (roi_center + roi_size / 2.0).astype(np.uint8)
    box_min = (roi_center - roi_size / 2.0).astype(np.uint16)
    box_max = (roi_center + roi_size / 2.0).astype(np.uint16)
    x_coords, y_coords ,z_coords= np.mgrid[
        box_min[0] : box_max[0] + 1,
        box_min[1] : box_max[1] + 1,
        box_min[2] : box_max[2] + 1
    ]
    return np.vstack([x_coords.ravel(), y_coords.ravel() ,z_coords.ravel()])

def get_circle_roi(image_shape,roi_center,roi_radius): #获得圆形区域的roi坐标
    Nz, Nx, Ny = image_shape
    center_z, center_x, center_y = roi_center
    z_coords, x_coords, y_coords = np.mgrid[0:Nz, 0:Nx, 0:Ny]
    
    #计算每个体素到圆心在 XY 平面上的距离平方
    distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2 + (z_coords - center_z)**2
    radius_sq = roi_radius**2
    mask = distance_sq <= radius_sq

    mask = mask.astype(np.uint8)
    z,x,y = np.where(mask>0)
    return np.vstack([z,x,y])

def get_roi_proj(image,geo,angles,valid_mesh):
    # 输入的mesh的shape要是(3,num_of_points)
    roi_image = np.zeros_like(image,np.float32)
    z_coords, x_coords ,y_coords = valid_mesh[0],valid_mesh[1],valid_mesh[2]
    roi_image[z_coords, x_coords, y_coords] = 1 #创建一个只有ROI区域有值的体膜
    roi_mask = tigre.Ax(roi_image,geo,angles)
    proj_data = tigre.Ax(image,geo,angles)
    roi_mask[roi_mask>0.01] = 1 #将其转换为mask
    roi_mask[roi_mask<=0.01] = 0
    roi_proj = proj_data * roi_mask #逐元素相乘提取数据
    return roi_proj,proj_data


## 找到满足要求的动脉区域（最有可能发生脑梗塞的区域）
def find_counterpart(mask,main_artery_indices):
    mask = np.round(mask).astype(np.uint8)
    mask_i = np.isin(mask, main_artery_indices)
    mask_filtered = np.where(mask_i, mask, 0)
    return mask_filtered

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

def sample_on_original_points_pca(points: np.ndarray, num_samples: int = 10):
    """
    通过 PCA 找到最长轴，将点云排序后，沿着该轴均匀选择原始点云上的点。
    
    参数:
        points: np.ndarray (N, 3) 原始点云坐标
        num_samples: int，采样数量
    返回:
        samples: np.ndarray (num_samples, 3) 选自原始点云的坐标
    """
    if len(points) < num_samples:
        return points # 点太少，返回全部

    # 1. PCA 分解 (与原代码相同)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    main_dir = eigvecs[:, 0]

    # 2. 投影到主轴，并根据投影值排序
    projections = centered @ main_dir
    # 获取排序后的索引：沿着主轴从小到大排列
    sorted_indices = np.argsort(projections)

    # 3. 均匀采样索引
    total_points = len(points)
    
    # 确定采样间隔
    interval = total_points / num_samples
    
    # 确定一个随机起始偏移量 (确保随机性)
    start_offset = np.random.uniform(0, interval)

    # 生成 num_samples 个均匀采样的浮点索引
    floating_indices = np.arange(num_samples) * interval + start_offset
    
    # 将浮点索引转换为整数索引 (并取模确保不越界)
    sample_indices = np.floor(floating_indices).astype(np.int32) % total_points

    # 4. 从排序后的索引中取出采样的点
    # 选中排序后列表中的这些点
    selected_indices_in_original = sorted_indices[sample_indices]
    
    # 返回原始坐标
    samples_on_original = points[selected_indices_in_original]
    
    return samples_on_original

def extract_centerlines(main_artery_mask, min_length=10):
    """
    对三维二值掩码进行骨架化，并分离出多条独立的中心线。

    参数:
        main_artery_mask (np.ndarray): 主血管的三维布尔/二值掩码 (D, H, W)。
        min_length (int): 过滤掉长度小于此值的短小中心线段。

    返回:
        centerlines_list (list): 包含每条独立中心线坐标点的列表。
        skeleton_labeled (np.ndarray): 带有标签的中心线三维图像 (D, H, W)。
    """
    
    if main_artery_mask.ndim != 3:
        raise ValueError("输入掩码必须是三维数组。")
    skeleton = skeletonize_3d(main_artery_mask)
    skeleton_labeled, num_labels = label(skeleton, return_num=True)
    centerlines_list = []
    for region in regionprops(skeleton_labeled):
        if region.area >= min_length:
            coords = region.coords
            centerlines_list.append(coords)
    return centerlines_list, skeleton_labeled

def get_roi(mask, sample_point, radius=50, angles = np.linspace(0,2*np.pi,360), geo = None):
    """
    mask: 图像的mask
    sample_point: 一个中心点
    para_path: 图像的参数的文件
    radius: 采样圆的半径
    """
    ## 处理单个对象，得到其截断图像（非截断区域设置为0）

    ## 设置几何
    # x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = read_para(para_path)
    # geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
    #             dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
    #             nDetector=np.array([1024,1024]),
    #             dDetector=np.array([0.5,0.5]))

    # for index,point in enumerate(sample_points):
    roi_center = np.array([sample_point[0], sample_point[1], sample_point[2]])
    ## 这里的image应该是mask
    point_mesh = get_circle_roi(mask.shape,roi_center,radius) # (3, N)的点云形式 #真值

    ## 进行投影
    roi_proj,_ = get_roi_proj(mask,geo,angles,point_mesh) #这个函数是通过点的坐标去构建三维体素，因此就算传入的mask是(x, y, z)的shape也没关系

    ## 将ROI的点云形式转换为体素形式的mask
    z_dim_max, x_dim_max, y_dim_max = geo.nVoxel[0], geo.nVoxel[1], geo.nVoxel[2], 
    mask_mask = np.zeros([z_dim_max, x_dim_max, y_dim_max], dtype=bool)
    for i in point_mesh.T:
        z_idx = int(round(i[0]))
        x_idx = int(round(i[1]))
        y_idx = int(round(i[2]))
        mask_mask[z_idx, x_idx, y_idx] = True
    roi_gt = np.where(mask_mask, mask, 0)

    return roi_proj,roi_gt

# def read_para(para_path):
#     # 读取几何参数
#     with open(para_path, 'r', encoding='utf-8') as f:
#         loaded_data = json.load(f)
#         x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = loaded_data['x_voxel_size'], loaded_data['y_voxel_size'], loaded_data['z_voxel_size'], loaded_data['x_dim_max'], loaded_data['y_dim_max'], loaded_data['z_dim_max']
#     return float(x_voxel_size), float(y_voxel_size), float(z_voxel_size), int(x_dim_max), int(y_dim_max), int(z_dim_max)


# 下面这个work的思路是直接得到真实图像和训练图像，但是应该是先得到完整的投影和图像，再选择角度去反投影。最后一步才是裁剪，
def get_sample_proj_vol_costa(mask_dir, artery_mask_dir, save_dir = './', number_roi_in_one_sample = 10, radius=60, already_has = None):
    # 设置保存路径
    centerline_save_path = join(save_dir,'centerline')
    point_cloud_save_path = join(save_dir,'point_cloud')
    print(centerline_save_path)
    os.makedirs(centerline_save_path, exist_ok=True)
    os.makedirs(point_cloud_save_path, exist_ok=True)
    print(point_cloud_save_path)
    # if(save_file):
    # 如果需要保留这些中间文件
    vol_gt_save_path = join(save_dir,'vol_gt')
    proj_gt_save_path = join(save_dir,'proj_gt')
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(proj_gt_save_path, exist_ok=True)

    # 其他参数
    angles = np.linspace(0,2*np.pi,360)
    radius = radius

    ## 对每个mask，都在主动脉区域选取10个点，并截取以这个点为圆心的圆形区域作为训练数据
    # num = 0
    for i in os.listdir(mask_dir):
        # 取得主要动脉的中心线
        if(already_has is not None and i in already_has):
            print('already: ', i)
            continue
        nib_artery_mask = nib.load(join(artery_mask_dir, i))
        nib_mask = nib.load(join(mask_dir,i))
        mask = nib_mask.get_fdata()
        artery_mask = nib_artery_mask.get_fdata()
        artery_mask = np.transpose(artery_mask, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        mask = mask.astype(np.float32)

        # main_artery_mask = find_counterpart(artery_mask, main_artery_indices) #把所有索引在main_artery_indices里面的部分提取出来，并返回一个ndarray的mask，因此artery_mask就相当于这个结果了
        _, skeleton_labeled = extract_centerlines(artery_mask) #这里的
        # tiff.imwrite(join(centerline_save_path,i),skeleton_labeled.astype(np.uint8)) #保留中心线文件

        # 在这些中心线上均匀且随机的取10个点
        point_cloud = np.vstack([np.where(skeleton_labeled>0)]).T #skeleton_labeled 来自于mask, 因此point_cloud的点坐标是(z,x,y), 因此sample_points的点坐标也是(z,x,y)
        sample_points = sample_on_original_points_pca(point_cloud, num_samples=number_roi_in_one_sample)
        
        # 保存选取的点
        convert_point_cloud_to_ply(sample_points,join(point_cloud_save_path,i.split('.')[0]+'.ply'))

        # 以点为中心取ROI区域
        mask[mask>0] = 1 #将mask置为0-12
        
        # 首先将roi区域内的mask内容提取出来，作为图像的真值
        for sample_index, sample_point in enumerate(sample_points):
            # 获得该中心点对应的roi
            x_voxel_size, y_voxel_size, z_voxel_size = [round(i, 4) for i in  nib_mask.header['pixdim'][1:4]]
            
            # nib文件加载的数据的shape是(x, y, z)
            z_dim_max, x_dim_max, y_dim_max = mask.shape[0], mask.shape[1], mask.shape[2]
            geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                nDetector=np.array([1024,1024]),
                dDetector=np.array([0.5,0.5]))
            roi_proj, roi_gt = get_roi(mask=mask, sample_point=sample_point, angles=angles, radius=radius, geo=geo) #与原始投影、原始图像的大小是一样的，但是只有ROI的区域有值，其他地方都是0

            # 把roi_gt从(z, x, y)变为(x, y, z)
            roi_gt = np.transpose(roi_gt, [1, 2, 0])
            roi_gt_nib  = nib.Nifti1Image(
                dataobj=roi_gt,
                affine=nib_mask.affine,
                header=nib_mask.header
            )

            save_name = f"{i.split('.')[0]}_sample_{str(sample_index).zfill(2)}"
            nib.save(roi_gt_nib, join(vol_gt_save_path,save_name+'.nii.gz'))
            # tiff.imwrite(join(proj_gt_save_path,save_name+'.tif'), roi_proj)
            np.savez_compressed(join(proj_gt_save_path,save_name+'.npz'),data=roi_proj)

        print(i)
        # break
        # num += 1
        # if(num==2):
        #     break

def read_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print(f"警告: 文件 {ply_path} 中没有读取到任何点。")
        return pcd, np.array([])
        
    # 将 Open3D 点云对象转换为 NumPy 数组
    point_cloud_np = np.asarray(pcd.points)
    return  point_cloud_np


def convert_point_cloud_to_ply(point_cloud,file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__":
    ## 对于这段代码，确实不需要more_radius, 这个值是后面在截取truncate图像的时候才会用到的。这里只是生成给定半径下的roi投影以及重建图像
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius",type=int)
    parser.add_argument("--number_roi_in_one_sample", type=int, default=10)
    parser.add_argument("--intensity", type=int, default=2)
    args = parser.parse_args()
    radius = args.radius
    number_roi_in_one_sample = args.number_roi_in_one_sample
    intensity = args.intensity
    ## 对于这段代码，确实不需要more_radius, 这个值是后面在截取truncate图像的时候才会用到的。这里只是生成给定半径下的roi投影以及重建图像
    
    already_has_proceed  =None

    ## 训练数据
    # print('处理训练数据')
    # mask_dir = '/data1/liuyang/stent/interior/data/costa/root_data/train_data'
    # save_dir = f'/data1/liuyang/stent/interior/data/costa/src_data/radius-{radius}/train_data'
    # # os.makedirs(save_dir, exist_ok=True)
    # artery_mask_dir = f'/data1/liuyang/stent/interior/data/costa/main_artery_mask_dir/train_data/{intensity}' #存储了主要动脉的mask的文件
    # # already_has_proceed = set([i.split("_sample_")[0]+'.nii.gz' for i in os.listdir(join(save_dir, 'vol_gt'))])
    # get_sample_proj_vol_costa(mask_dir=mask_dir, artery_mask_dir=artery_mask_dir, save_dir=save_dir, number_roi_in_one_sample=number_roi_in_one_sample, radius=radius, already_has = already_has_proceed)

    # 测试数据
    print('处理测试数据')
    mask_dir = '/data1/liuyang/stent/interior/data/costa/root_data/test_data'
    save_dir = f'/data1/liuyang/stent/interior/data/costa/src_data/radius-{radius}/test_data'
    # os.makedirs(save_dir, exist_ok=True)
    artery_mask_dir = f'/data1/liuyang/stent/interior/data/costa/main_artery_mask_dir/test_data/{intensity}' #存储了主要动脉的mask的文件
    # already_has_proceed = set([i.split("_sample_")[0]+'.nii.gz' for i in os.listdir(join(save_dir, 'vol_gt'))])
    get_sample_proj_vol_costa(mask_dir=mask_dir, artery_mask_dir=artery_mask_dir, save_dir=save_dir, number_roi_in_one_sample=number_roi_in_one_sample, radius=radius, already_has = already_has_proceed)


