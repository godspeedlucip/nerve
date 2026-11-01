import os
import numpy as np
import plyfile
import tifffile as tiff
join = os.path.join
import matplotlib.pyplot as plt
import shutil
import open3d as o3d
from matplotlib.colors import Normalize
import SimpleITK as sitk

from geomdl import NURBS, utilities
from scipy.ndimage import convolve, gaussian_filter
import tigre

import open3d as o3d

import random

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


def center_point_cloud_to_voxel_space(
    point_cloud: np.ndarray, 
    voxel_size: tuple = (128, 128, 256)
) -> np.ndarray:
    """
    将点云 (N, 3) 移动到以 voxel_size 定义的空间的中心位置。

    参数:
    - point_cloud: 形状为 (N, 3) 的 NumPy 数组 (点云)。
    - voxel_size: 目标体素空间的尺寸 (Dx, Dy, Dz)。默认为 (128, 128, 256)。

    返回:
    - centered_point_cloud: 居中后的点云 (N, 3) NumPy 数组。
    """
    
    # 1. 计算点云的中心点 (质心)
    # np.mean(axis=0) 计算每列的平均值 (X, Y, Z 的平均值)
    curve_center = np.mean(point_cloud, axis=0)
    
    # 2. 计算目标体素空间的中心点
    target_center = np.array([dim / 2.0 for dim in voxel_size], dtype=np.float32)
    
    # 3. 计算平移向量
    # 平移向量 = 目标中心 - 曲线中心
    translation_vector = target_center - curve_center
    
    # 4. 应用平移
    # NumPy 的广播机制会自动将 (3,) 的向量加到 (N, 3) 的数组的每一行上
    centered_point_cloud = point_cloud + translation_vector
    
    return centered_point_cloud


def rotate_point_cloud(
    point_cloud: np.ndarray, 
    axis = 0,
    angle = 90
) -> np.ndarray:
    """
    点云绕 X, Y, 或 Z 轴旋转任意角度。

    参数:
    - point_cloud: 形状为 (N, 3) 的点云数组。
    - axis: 旋转轴，必须是 'X', 'Y', 或 'Z' (不区分大小写)。
    - angle_degrees: 旋转角度 (度数)。

    返回:
    - rotated_points: 旋转后的点云 (N, 3)。
    """
    
    # 1. 转换角度为弧度
    theta = np.deg2rad(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    axises = ["X",'Y','Z']
    axis = axises[axis]
    # 2. 定义旋转矩阵 R (3x3)
    axis = axis.upper()
    
    if axis == 'X': # X
        R = np.array([
            [1, 0, 0],
            [0, cos_t, -sin_t],
            [0, sin_t, cos_t]
        ],dtype=np.float32)
    elif axis == 'Y': # Y
        R = np.array([ 
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ],dtype=np.float32)
    elif axis == 'Z': # Z
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ],dtype=np.float32)
    else:
        raise ValueError("旋转轴必须是 X, Y, 或 Z")
    
    # 3. 应用旋转
    # 点云 P (N, 3) 乘以旋转矩阵 R (3, 3) 得到 P_new (N, 3)
    # 结果：P_new = P @ R
    rotated_points = point_cloud @ R
    
    return rotated_points


def compute_curve_length(point_cloud):
    """
    计算点云曲线的长度
    """
    diffs = np.diff(point_cloud, axis=0)  # 相邻点差
    seg_lengths = np.linalg.norm(diffs, axis=1)  # 每段线段长度
    return np.sum(seg_lengths)

def rescale_point_cloud(point_cloud, target_length, center_mode="start"):
    """
    如果点云曲线长度超过 target_length，则进行等比例缩小
    center_mode: "start" -> 从起点缩放, "center" -> 从几何中心缩放
    """
    # 计算原始长度
    length = compute_curve_length(point_cloud)

    if length <= target_length:
        return point_cloud  # 不需要缩放

    # 缩放比例
    scale = target_length / length

    # 确定缩放中心
    if center_mode == "start":
        center = point_cloud[0]  # 以起点为缩放基准
    elif center_mode == "center":
        center = np.mean(point_cloud, axis=0)  # 几何中心
    else:
        raise ValueError("center_mode 必须是 'start' 或 'center'")

    # 平移到中心 → 缩放 → 平移回去
    shifted = point_cloud - center
    scaled = shifted * scale
    rescaled_cloud = scaled + center

    return rescaled_cloud


def stent_simu(pc):
    x_min = np.min(pc[:,0])
    pc[:,0] = pc[:,0] - x_min
    x_max = np.max(pc[:,0])
    y_min = np.min(pc[:,1])
    pc[:,1] = pc[:,1] - y_min
    y_max = np.max(pc[:,1])
    pc[:,0] = pc[:,0] - (x_max)/2
    pc[:,1] = pc[:,1] - (y_max)/2
    return pc


def tilt_point_cloud(point_cloud, angle_deg, axis=0, center_mode='center', boundary_clearance=18, ndims=(128,128,256),clip=True):
    """
    将给定点云整体倾斜（绕某个轴旋转指定角度）。
    
    point_cloud: (N,3) 点云
    angle_deg: 倾斜角度（度）
    axis: 'x','y','z'
    center_mode: 'center' 或 'start'，旋转中心
    boundary_clearance: int, 如果提供，则旋转后保证离边界至少 n
    ndims: (X_MAX,Y_MAX,Z_MAX), 必须提供才能进行边界限制
    """
    angle = np.deg2rad(angle_deg)
    axis = ['x','y','z'][axis]
    # 旋转矩阵
    if axis=='x':
        R = np.array([[1,0,0],
                      [0,np.cos(angle), -np.sin(angle)],
                      [0,np.sin(angle),  np.cos(angle)]])
    elif axis=='y':
        R = np.array([[ np.cos(angle),0,np.sin(angle)],
                      [0,1,0],
                      [-np.sin(angle),0,np.cos(angle)]])
    elif axis=='z':
        R = np.array([[np.cos(angle), -np.sin(angle),0],
                      [np.sin(angle),  np.cos(angle),0],
                      [0,0,1]])
    else:
        raise ValueError("axis 必须是 'x','y','z'")
    
    # 旋转中心
    if center_mode=='center':
        center = np.mean(point_cloud, axis=0)
    elif center_mode=='start':
        center = point_cloud[0]
    else:
        raise ValueError("center_mode 必须是 'center' 或 'start'")
    
    # 平移到中心 → 旋转 → 平移回去
    shifted = point_cloud - center
    rotated = shifted @ R.T
    tilted_cloud = rotated + center
    
    # 边界限制
    if(clip):
        if boundary_clearance is not None and ndims is not None:
            X_MAX,Y_MAX,Z_MAX = ndims
            tilted_cloud[:,0] = np.clip(tilted_cloud[:,0], boundary_clearance, X_MAX-1-boundary_clearance)
            tilted_cloud[:,1] = np.clip(tilted_cloud[:,1], boundary_clearance, Y_MAX-1-boundary_clearance)
            tilted_cloud[:,2] = np.clip(tilted_cloud[:,2], 0, Z_MAX-1)
    return tilted_cloud

def get_min_max(pc):
    return np.min(pc[:,0]),np.max(pc[:,0]),np.min(pc[:,1]),np.max(pc[:,1]),np.min(pc[:,2]),np.max(pc[:,2])

def flip_point_cloud(point_cloud: np.ndarray,axis = 2) -> np.ndarray:
    """
    对点云进行上下颠倒（Z轴镜像翻转），以其几何中心平面为对称面。
    
    参数:
    - point_cloud: (N, 3) 形状的点云数组 (x, y, z)。
    
    返回:
    - flipped_cloud: 翻转后的点云。
    """
    flipped_cloud = point_cloud.copy()
    z_center = np.mean(point_cloud[:, axis])
    flipped_cloud[:, axis] = 2 * z_center - flipped_cloud[:, axis]
    return flipped_cloud

def visualize_point_cloud(points_data):
    """
    可视化三维点云数据。

    Args:
        points_data (np.ndarray): 包含点x, y, z坐标的NumPy数组，形状为(N, 3)。
        filename (str, optional): 如果提供，将点云保存到文件中。
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 从 NumPy 数组赋值点云
    pcd.points = o3d.utility.Vector3dVector(points_data)
    
    # 可选：给点云上色，这里给所有点赋上统一的颜色
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="点云三维图",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50)

def visualize_braid_stent(points, voxel_size=(128,128,256)):
    """
    Draws a 3D scatter plot from a numpy array of points.
    
    Args:
        points (np.ndarray): A numpy array of shape (N, 3) representing N 3D points.
        filename (str): The filename to save the plot as.
    """
    if points.shape[0] == 0:
        print("Point array is empty, nothing to draw.")
        return

    # Create a new figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')

    ax.set_xlim(0, voxel_size[0])
    ax.set_ylim(0, voxel_size[1])
    ax.set_zlim(0, voxel_size[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Simulated Stent (Continuous Appearance)')
    ax.set_box_aspect([1, 1, 1])
    plt.show()

# # 5. 可视化
# def visualize_both(cyl_points,cyl_points_deformed,curve_points):
#     fig = plt.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(cyl_points[:,0], cyl_points[:,1], cyl_points[:,2], s=2, alpha=0.5)
#     ax1.set_title("Original Straight Cylinder")
#     ax1.set_box_aspect([1,1,1])
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.scatter(cyl_points_deformed[:,0], cyl_points_deformed[:,1], cyl_points_deformed[:,2], s=2, alpha=0.5, c="r")
#     ax2.plot(curve_points[0], curve_points[1], curve_points[2], "k--", label="Centerline")
#     ax2.set_title("Deformed Cylinder (Point Cloud)")
#     ax2.set_box_aspect([1,1,1])
#     plt.show()


def scale_point_cloud(point_cloud, voxel_size=(128,128,256)):
    """
    将点云等比例缩放到给定的体素空间。

    参数:
    point_cloud (np.ndarray): 输入点云，形状为 (N, 3)，N 是点的数量。
    voxel_shape (tuple): 体素空间的尺寸，例如 (Dx, Dy, Dz)。

    返回:
    np.ndarray: 缩放和平移后的点云，值在 [0, D-1] 范围内。
    """
    # 1. 计算点云的边界框 (min 和 max)
    # 沿每个维度 (x, y, z) 找到最小值和最大值
    min_coords = np.min(point_cloud, axis=0)  # (3,) 
    max_coords = np.max(point_cloud, axis=0)  # (3,)

    # 2. 计算点云的范围 (range)
    # range = max - min
    cloud_range = max_coords - min_coords  # (3,)

    # 3. 计算最大的范围维度，以确定统一的缩放比例
    # 确保等比例缩放
    max_range = np.max(cloud_range) 
    
    # 4. 计算缩放比例 (scale factor)
    # 目标体素空间的最大尺寸 - 1 (因为体素索引从 0 开始)
    # 这里我们假设目标空间是 (128, 128, 256) 这种，最大尺寸是 256
    # 体素空间的尺寸，例如 (128, 128, 256)
    voxel_dim = np.array(voxel_size)
    # 目标空间最大索引，例如 256 - 1 = 255
    max_voxel_index = np.max(voxel_dim) - 1 

    # 缩放比例: (目标空间的最大索引) / (点云的最大范围)
    # 如果 max_range 为 0 (所有点重合)，则 scale_factor 设为 1 或其他合理值
    if max_range == 0:
        scale_factor = 1.0
    else:
        scale_factor = max_voxel_index / max_range

    # 5. 平移点云
    # 将点云的最小点平移到原点 (0, 0, 0)
    translated_cloud = point_cloud - min_coords

    # 6. 缩放点云
    # 等比例缩放，保持几何形状不变
    scaled_cloud = translated_cloud * scale_factor

    # 7. 确保点云在体素空间内 (可选，用于最终校验)
    # 理论上，scaled_cloud 的值在 [0, max_voxel_index] 范围内
    # 如果需要将浮点坐标转换为体素索引 (整数)
    # final_coords = np.round(scaled_cloud).astype(int)
    
    return scaled_cloud

def scale_point_axes(point_cloud,axes_value=[1,1,1]):
    # print(point_cloud.dtype)
    point = np.copy(point_cloud)
    point[:,0] *= axes_value[0]
    point[:,1] *= axes_value[1]
    point[:,2] *= axes_value[2]
    return point

def convert_3d_array_to_ply(array: np.ndarray,plyfile=False):
    """
    将一个形状为 (300, 102, 102) 的 NumPy 数组转换为 PLY 格式的点云文件。

    该函数将数组中值大于 0 的元素视为有效的点，并将其坐标转换为 PLY 文件的
    (x, y, z) 格式。

    参数:
    array (np.ndarray): 输入的 NumPy 数组，形状应为 (300, 102, 102)。
    filename (str): 输出的 PLY 文件的路径和名称。
    """

    # 找到所有非零元素的索引
    x,y,z= np.where(array > 0)
    points = np.vstack((x, y, z)).T
    
    if(plyfile):
        vertex_data = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        el = plyfile.PlyElement.describe(vertex_data, 'vertex')
        ply_file = plyfile.PlyData([el], text=False)
        return points,ply_file
    else:
        return points

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

def voxelize_1(stent_points,voxel_size = (128,128,256),fill_value = 1):
    voxel = np.zeros(voxel_size,dtype=np.float32)

    for i in stent_points:
        x = int(np.round(i[0]))
        y = int(np.round(i[1]))
        z = int(np.round(i[2]))
        if(x>=0 and x<voxel_size[0] and y>=0 and y<voxel_size[1] and z>=0 and z<voxel_size[2]):
            ## 确保在界内
            voxel[x][y][z] = fill_value
    return voxel


def voxelize_2(stent_points,voxel_size = (128,128,256),fill_value = 1):
    voxel = np.zeros(voxel_size,dtype=np.float32)

    for i in stent_points:
        x = int(np.round(i[0]))
        y = int(np.round(i[1]))
        z = int(np.round(i[2]))
        if_inside = True
        if(x>=0 and x<voxel_size[0] and y>=0 and y<voxel_size[1] and z>=0 and z<voxel_size[2]):
            ## 确保在界内
            voxel[x][y][z] = fill_value
        else:
            if(if_inside):
                if_inside = False
    return voxel,if_inside

# def gau(point_cloud):
#     ## 将其体素化
#     deform_voxel = voxelize_1(point_cloud)

#     ## 对其进行平滑处理
#     gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.7,kernel_size=2,valid_thred=2)

#     return gau
