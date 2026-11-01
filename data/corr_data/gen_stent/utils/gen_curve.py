import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tifffile as tiff
join = os.path.join
import matplotlib.pyplot as plt
import shutil
import open3d as o3d
from matplotlib.colors import Normalize
import SimpleITK as sitk
from geomdl import NURBS, utilities
import open3d as o3d
import random


from .base_utils import tilt_point_cloud, flip_point_cloud, center_point_cloud_to_voxel_space, rescale_point_cloud
from .file_rw import convert_point_cloud_to_ply



## 生成阿基米德螺旋线
def generate_constrained_spiral_curve_iterative(
    target_length: float, 
    num_points: int = 1000,
    voxel_size: tuple = (128, 128, 256),
    xy_padding: float = 18.0,
    z_min_span: float = 128.0,
    max_iterations: int = 50,
    tolerance: float = 1.0  # 允许与目标距离的误差 (像素)
) -> np.ndarray:
    """
    通过迭代方式生成满足所有约束（长度、Z轴跨度、X/Y边界距离）的曲线。

    参数:
    - target_length: 目标曲线离散长度 (float)。
    - num_points: 最终点云中的点数 N (int)。
    - voxel_size: (x_max, y_max, z_max) 空间尺寸的元组 (tuple)。
    - xy_padding: X, Y 坐标距离边界的最小距离。
    - z_min_span: Z 轴跨度的最小要求。
    - max_iterations: 最大迭代次数。
    - tolerance: 约束检查的容忍度。

    返回:
    - final_curve_points: 形状为 (N, 3) 的 NumPy 数组。
    """
    
    # --- 初始曲线形状参数 ---
    # 调整这些参数以得到一个接近目标的初始形状
    T_max = 1.2      # 控制整体长度和圈数
    A = 5.0          # 半径增长速度 (影响 XY 跨度)
    B = 15          # 初始半径
    k1 = 10.0        # 旋转频率
    C = 5.0          # Z 轴拉伸速度 (影响 Z 跨度)

    # 目标 XY 轴的最大允许跨度
    MAX_XY_SPAN = voxel_size[0] - 2 * xy_padding

    for iteration in range(max_iterations):
        
        # 1. 生成基于当前参数的曲线形状
        t = np.linspace(0, T_max, num_points)
        radius = A * t + B
        x = radius * np.cos(k1 * t)
        y = radius * np.sin(k1 * t)
        z = C * t
        
        current_points = np.stack([x, y, z], axis=1)
        start_point = current_points[0, :]
        
        # 2. 几何检查和长度计算
        segment_lengths = np.sqrt(np.sum(np.diff(current_points, axis=0)**2, axis=1))
        current_length = np.sum(segment_lengths)
        initial_z_span = np.max(current_points[:, 2]) - np.min(current_points[:, 2])
        
        # 检查 X/Y 轴的原始跨度 (用于迭代调整 A)
        initial_x_span = np.max(current_points[:, 0]) - np.min(current_points[:, 0])
        initial_y_span = np.max(current_points[:, 1]) - np.min(current_points[:, 1])

        # 3. 约束检查
        
        # 检查 Z 轴跨度是否满足要求
        z_span_ok = initial_z_span >= z_min_span - tolerance
        
        # 检查 X/Y 轴跨度是否满足安全边距 (即是否小于 MAX_XY_SPAN)
        xy_span_ok = (initial_x_span <= MAX_XY_SPAN + tolerance) and \
                     (initial_y_span <= MAX_XY_SPAN + tolerance)
        
        
        # --- 4. 调整参数 ---
        
        if z_span_ok and xy_span_ok:
            # print(f"迭代 {iteration + 1}: 所有约束均满足。")
            break  # 满足所有约束，跳出循环

        # Z 轴跨度不足: 增加 Z 轴拉伸速度 C
        if not z_span_ok:
            # 增加 C 来提升 Z 轴跨度。调整步长与缺失的比例成正比。
            C *= (z_min_span / initial_z_span) ** 0.5 
            # print(f"迭代 {iteration + 1}: Z 跨度不足 ({initial_z_span:.2f} < {z_min_span:.1f})，调整 C={C:.2f}")

        # X/Y 跨度过大 (触及边界): 减小螺旋半径参数 A
        if not xy_span_ok:
            # 减小 A 来收缩 XY 跨度。取最坏情况 (最大的跨度) 来计算收缩比。
            max_span = max(initial_x_span, initial_y_span)
            A *= (MAX_XY_SPAN / max_span) ** 0.5
            # print(f"迭代 {iteration + 1}: XY 跨度过大 ({max_span:.2f} > {MAX_XY_SPAN:.1f})，调整 A={A:.2f}")

    # --- 5. 最终缩放和平移 (保证长度和居中) ---
    
    # 重新生成最终形状的点集
    t = np.linspace(0, T_max, num_points)
    radius = A * t + B
    x = radius * np.cos(k1 * t)
    y = radius * np.sin(k1 * t)
    z = C * t
    final_shape_points = np.stack([x, y, z], axis=1)
    
    # 计算最终形状的长度
    segment_lengths = np.sqrt(np.sum(np.diff(final_shape_points, axis=0)**2, axis=1))
    final_shape_length = np.sum(segment_lengths)
    
    # 长度缩放因子
    length_scale_factor = target_length / final_shape_length
    
    # 应用长度缩放 (相对于起点)
    final_start_point = final_shape_points[0, :]
    scaled_points = (final_shape_points - final_start_point) * length_scale_factor + final_start_point
    
    # 平移至体素空间中心
    min_coords = np.min(scaled_points, axis=0)
    max_coords = np.max(scaled_points, axis=0)
    curve_center = (min_coords + max_coords) / 2
    
    target_center = np.array([d / 2.0 for d in voxel_size])
    translation_vector = target_center - curve_center
    
    final_curve_points = scaled_points + translation_vector
    
    # --- 6. 最终验证和输出 ---
    
    final_length_check = np.sum(np.sqrt(np.sum(np.diff(final_curve_points, axis=0)**2, axis=1)))
    z_span_final = np.max(final_curve_points[:, 2]) - np.min(final_curve_points[:, 2])
    x_min, x_max = np.min(final_curve_points[:, 0]), np.max(final_curve_points[:, 0])
    y_min, y_max = np.min(final_curve_points[:, 1]), np.max(final_curve_points[:, 1])
    x_dist_to_boundary = min(x_min, voxel_size[0] - x_max)
    y_dist_to_boundary = min(y_min, voxel_size[1] - y_max)
    
    # print("-" * 30)
    # print("【最终结果验证】")
    # print(f"目标长度: {target_length:.2f} | 实际长度: {final_length_check:.2f}")
    # print(f"目标 Z 跨度 >= {z_min_span:.1f} | 实际 Z 跨度: {z_span_final:.2f}")
    # print(f"目标 XY 边距 >= {xy_padding:.1f} | X 边距: {x_dist_to_boundary:.2f} | Y 边距: {y_dist_to_boundary:.2f}")
    # print("-" * 30)

    # 警告：如果迭代没有收敛，输出一条警告
    # if iteration == max_iterations - 1:
    #     print("警告: 迭代达到最大次数，但可能未完全收敛到所有约束。")
    
    return final_curve_points


## 生成u型曲线
def create_outward_arc_u_shape_point_cloud(
    total_length: float = 300,
    height: float = 85, #圆弧的直径，不能过大，应该要小于 128 - 20 * 2（18是给支架的半径留的间距）
    total_points: int = 10000,
    z_coord: float = 0.0,
    flip_or_not = False,
) -> np.ndarray:
    
    """
    创建由两条长直线段和一个向外凸出圆弧连接而成的 U 形图形的离散点云 (N, 3)。
    
    参数:
        length (float): 平行直线段的长度。
        height (float): 平行直线段之间的距离（即圆弧的直径）。
        num_points_per_segment (int): 每条直线段和圆弧上的点数。
        z_coord (float): 曲线所在的固定 Z 坐标。

    返回:
        numpy.ndarray: 形状为 (N, 3) 的点云数组。
    """

    single_len = (total_length - np.pi * (height / 2)) / 2 #圆弧以外的部分，也就是两条直线。这就是每条直线的长度
    assert single_len > 0
    single_len_points = int(total_points * single_len / total_length) #单条直线的点的个数
    single_arc_points = total_points - 2 * single_len_points
    R = height / 2.0  # 圆弧半径 R = 0.5
    
    # --- 1. 下直线段 (长): 从 (0, 0) 到 (L, 0) ---
    x_bottom = np.linspace(0, single_len, single_len_points)
    y_bottom = np.zeros_like(x_bottom)
    
    # --- 2. 向外圆弧 (右侧): 从 (L, 0) 到 (L, H) ---
    # 圆心 C = (L, R) = (5.0, 0.5)
    center_x, center_y = single_len, R
    
    # 角度范围: 从-90度到90度，0度可以理解为x轴，90度可以理解为y轴的反方向
    start_rad = -np.pi * 0.5
    end_rad = np.pi * 0.5
    theta = np.linspace(start_rad, end_rad, single_arc_points)
    
    # 圆弧参数方程
    x_arc = center_x + R * np.cos(theta)
    y_arc = center_y + R * np.sin(theta)
    
    # --- 3. 上直线段 (长): 从 (L, H) 到 (0, H) ---
    # 逆向连接，以保持点云顺序连续
    x_top = np.linspace(single_len, 0, single_len_points)
    y_top = np.full_like(x_top, height)
    
    # --- 4. 组合所有段的 X, Y 坐标 ---
    # 顺序：下直线 -> 圆弧 -> 上直线 (反向)
    all_x = np.concatenate([x_bottom, x_arc, x_top])
    all_y = np.concatenate([y_bottom, y_arc, y_top])
    
    # 5. 生成 Z 坐标，并组合成 (N, 3) 数组
    all_z = np.full_like(all_x, z_coord)

    point_cloud = np.stack((all_z, all_y, all_x), axis=1) # 这样排列可以让U型曲线的开口朝向z轴的负方向
    
    if(flip_or_not): #如果需要翻转就翻转
        point_cloud = flip_point_cloud(point_cloud,axis=2)

    point_cloud = center_point_cloud_to_voxel_space(point_cloud,(128, 128, 256)) #将其移动到体素空间的中心

    rand_num = random.randint(0,2)
    if(rand_num % 3==2): #绕z轴旋转
        random_angle = np.random.rand() * 180
        point_cloud = tilt_point_cloud(point_cloud,axis= 2, angle_deg = random_angle) #绕z轴旋转
    elif(rand_num % 3==1): #绕y轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 1, angle_deg = random_angle) #绕y轴旋转
    else: #绕x轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 0, angle_deg = random_angle) #绕y轴旋转

    # 再移动到体素空间中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud,(128, 128, 256)) #将其移动到体素空间的中心

    return point_cloud

# u_shape_points = create_outward_arc_u_shape_point_cloud(total_length=300, height=80, total_points=20000,flip_or_not=True)
# visualize_braid_stent(u_shape_points)
# visualize_point_cloud(u_shape_points)



## 生成单正弦曲线
def generate_single_sin_curve(
    voxel_size = (128,128,256),
    N=30000, 
    clip_interval=20,
    z_clip_interval=10,
    total_length = 256,
    amplitude=None,
    wavelength=None,
    rate = 0.5
):
    """
    在体素空间中生成一条垂直于XY平面的正弦曲线，并返回点云形式。
    """

    MIN_COORD = clip_interval
    MAX_COORD_X = voxel_size[0] - 1 - clip_interval
    MAX_COORD_Y = voxel_size[1] - 1 - clip_interval

    # 振幅：默认取 XY 平面内的安全范围一半
    if amplitude is None:
        amplitude = min((MAX_COORD_X - MIN_COORD) / 2,
                        (MAX_COORD_Y - MIN_COORD) / 2)
    
    z_min = z_clip_interval
    z_max = voxel_size[2] - z_clip_interval - 1

    # 波长：默认等于整个 z 范围（完整一个周期）
    if wavelength is None:
        wavelength = rate * (z_max - z_min)

    # --- 2. 生成 z 坐标 ---
    z_coords = np.linspace(z_min, z_max, N)

    # --- 3. 正弦波 (在 x 方向摆动, y 固定) ---
    x_center = (MIN_COORD + MAX_COORD_X) // 2
    y_center = (MIN_COORD + MAX_COORD_Y) // 2

    # 正弦函数：x = center + A * sin(2π z / wavelength)
    x_coords = x_center + amplitude * np.sin(2 * np.pi * z_coords / wavelength)
    y_coords = np.full(N, y_center)

    # --- 4. 组合点云 ---
    point_cloud = np.stack((x_coords, y_coords, z_coords), axis=1)

    # # 进行放缩
    point_cloud = rescale_point_cloud(point_cloud,total_length)

    # 移动到体素空间的中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud)

    ## 进行旋转
    # 绕z轴旋转是必须的
    random_angle = np.random.rand() * 180
    point_cloud = tilt_point_cloud(point_cloud,axis= 2, angle_deg = random_angle,clip=False) 

    ## 绕xy轴旋转是可选的
    rand_num = random.randint(0,1)
    if(rand_num==1): #绕y轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 1, angle_deg = random_angle,clip=False) #绕y轴旋转
    else: #绕x轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 0, angle_deg = random_angle,clip=False) #绕y轴旋转

    # # 再移动到体素空间中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud)


    ## 进行裁剪
    point_cloud[:,0] = np.clip(point_cloud[:,0], clip_interval, voxel_size[0]-1-clip_interval)
    point_cloud[:,1] = np.clip(point_cloud[:,1], clip_interval, voxel_size[1]-1-clip_interval)
    point_cloud[:,2] = np.clip(point_cloud[:,2], 0, voxel_size[2] - z_clip_interval-1)

    return point_cloud

# 示例调用
# point_cloud = generate_single_sin_curve(N=10000, total_length=350, voxel_size=(128,128,256),rate=0.5)
# visualize_braid_stent(point_cloud)
# visualize_point_cloud(point_cloud)





## 生成双正弦曲线

def generate_3d_sine_curve_pointcloud(
    voxel_size=(128,128,256),
    N=50000, 
    clip_interval=20,
    z_clip_interval=10,
    wavelength_x=None, wavelength_y=None,
    phase_x=0.0, phase_y=0.0,
    total_length = 350
):
    """
    在体素空间中生成一条 3D 正弦曲线 (x,z 和 y,z 上都正弦摆动)，返回点云形式。
    """

    # --- 1. 参数与边界 ---

    MIN_COORD = clip_interval
    MAX_COORD_X = voxel_size[0] - 1 - clip_interval
    MAX_COORD_Y = voxel_size[0] - 1 - clip_interval

    # 中心点
    x_center = (MIN_COORD + MAX_COORD_X) // 2
    y_center = (MIN_COORD + MAX_COORD_Y) // 2

    amplitude_x = (MAX_COORD_X - MIN_COORD) / 2
    amplitude_y = (MAX_COORD_Y - MIN_COORD) / 2


    z_min = z_clip_interval
    z_max = voxel_size[2] - z_clip_interval
    # 默认波长：整个 z 范围一周期
    if(wavelength_x is None):
        wavelength_x = (z_max - z_min)
    if(wavelength_y is None):
        wavelength_y = (z_max - z_min)

    # --- 2. z 坐标 ---
    z_coords = np.linspace(z_min, z_max, N)

    # --- 3. x,y 正弦摆动 ---
    x_coords = x_center + amplitude_x * np.sin(2 * np.pi * z_coords / wavelength_x + phase_x)
    y_coords = y_center + amplitude_y * np.sin(2 * np.pi * z_coords / wavelength_y + phase_y)

    # --- 5. 点云组合 ---
    point_cloud = np.stack((x_coords, y_coords, z_coords), axis=1)

    point_cloud = rescale_point_cloud(point_cloud,total_length) # 缩放达到指定长度

    ## 移动到体素空间中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud,voxel_size=(128,128,256))

    # 绕z轴旋转是必须的
    random_angle = np.random.rand() * 180
    point_cloud = tilt_point_cloud(point_cloud,axis= 2, angle_deg = random_angle,clip=False) 

    ## 绕xy轴旋转是可选的
    rand_num = random.randint(0,1)
    if(rand_num==1): #绕y轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 1, angle_deg = random_angle,clip=False) #绕y轴旋转
    else: #绕x轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 0, angle_deg = random_angle,clip=False) #绕y轴旋转

    ## 再次移动到体素空间的中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud,voxel_size=(128,128,256))

    ## 裁剪
    point_cloud[:,0] = np.clip(point_cloud[:,0], clip_interval, voxel_size[0]-1-clip_interval)
    point_cloud[:,1] = np.clip(point_cloud[:,1], clip_interval, voxel_size[1]-1-clip_interval)
    point_cloud[:,2] = np.clip(point_cloud[:,2], 0, voxel_size[2] - z_clip_interval-1)


    return point_cloud


## 生成随机曲线

def generate_random_smooth_curve(
    voxel_size=(128,128,512),
    curvature_range=(0.6, 0.7),
    num_points=20000,
    clip_interval = 20,
    z_clip_interval = 5,
    num_coeffs = 5
):
    """
    生成一条在指定空间内、z轴上没有交叉、具有随机且更平滑弧度的非螺旋曲线点云。

    核心改进：使用 5 阶多项式来提高平滑度，并增加采样密度。
    
    Args:
        space_x (float): X轴的最大范围。
        space_y (float): Y轴的最大范围。
        space_z (float): Z轴的最大范围。
        curvature_range (tuple): 控制弯曲程度的因子范围，例如 (0.25, 0.50) 表示最大偏移在总空间的 25% 到 50%。
        num_points (int, optional): 点云中的点数。如果为 None, 则随机选择。

    Returns:
        np.ndarray: 形状为 (N, 3) 的点云数组。
    """
    
    # --- 1. 基础参数设置 ---
    t_min = 0.0 + z_clip_interval
    t_max = voxel_size[2] - z_clip_interval
    
    # 增加默认点数，确保采样平滑
    if num_points is None:
        N = np.random.randint(500, 1000) 
    else:
        N = num_points
        
    # --- 2. 弧度控制参数 ---
    max_xy_dim = min(voxel_size[0], voxel_size[1])
    
    # 根据用户传入的范围控制弯曲程度
    A_scale = np.random.uniform(max_xy_dim * curvature_range[0], max_xy_dim * curvature_range[1]) 
    
    # 曲线在 (X, Y) 平面上的起始点偏移
    start_offset_x = np.random.uniform(0.1 * voxel_size[0], 0.9 * voxel_size[0])
    start_offset_y = np.random.uniform(0.1 * voxel_size[1], 0.9 * voxel_size[1])

    # --- 3. 随机生成 5 阶多项式基础系数 (需要 6 个系数) ---
    # 系数范围可以保持在 [-1.0, 1.0]，依靠 A_scale 进行整体缩放。
    num_coeffs = num_coeffs # 对应 4 阶多项式
    coeffs_x = np.random.uniform(-1.0, 1.0, num_coeffs)
    coeffs_y = np.random.uniform(-1.0, 1.0, num_coeffs)
    
    # --- 4. 生成参数 t 数组并归一化 ---
    t = np.linspace(t_min, t_max, N)
    t_norm = 2 * (t - t_min) / (t_max - t_min) - 1 
    
    # --- 5. 计算坐标并进行空间映射 ---
    
    # 使用 5 阶多项式计算基础值
    x_base = np.polyval(coeffs_x, t_norm)
    y_base = np.polyval(coeffs_y, t_norm)
    
    # 应用缩放和平移
    x = x_base * A_scale + start_offset_x
    y = y_base * A_scale + start_offset_y
    z = t # Z轴单调性保证
    point_cloud = np.stack((x, y, z), axis=-1)

    # # 进行放缩
    point_cloud = rescale_point_cloud(point_cloud,256)

    # 移动到体素空间的中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud)

    ## 进行旋转
    # 绕z轴旋转是必须的
    random_angle = np.random.rand() * 180
    point_cloud = tilt_point_cloud(point_cloud,axis= 2, angle_deg = random_angle,clip=False) 

    ## 绕xy轴旋转是可选的
    rand_num = random.randint(0,1)
    if(rand_num==1): #绕y轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 1, angle_deg = random_angle,clip=False) #绕y轴旋转
    else: #绕x轴旋转
        random_angle = np.random.rand() * 45
        point_cloud = tilt_point_cloud(point_cloud,axis= 0, angle_deg = random_angle,clip=False) #绕y轴旋转

    # # 再移动到体素空间中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud)

    ## 进行裁剪
    point_cloud[:,0] = np.clip(point_cloud[:,0], clip_interval, voxel_size[0]-1-clip_interval)
    point_cloud[:,1] = np.clip(point_cloud[:,1], clip_interval, voxel_size[1]-1-clip_interval)
    point_cloud[:,2] = np.clip(point_cloud[:,2], 0, voxel_size[2] - z_clip_interval-1)

    
    return point_cloud

# # 示例调用 (使用默认空间和中等弯曲)
# point_cloud_data = generate_smooth_curve_improved()

# # visualize_point_cloud(point_cloud_data)
# visualize_braid_stent(point_cloud_data)



## 生成NURBS随机曲线
def create_nurbs_centerline_fixed_length(num_control_points, voxel_size=(128,128,256), target_length=100,num_eval_points=10000,clip_interval=20): #与边界至少保留20个像素
    '''
    返回值: 在生成的曲线上均匀采样的num_eval_points个点
    '''

    def curve_length(points):
        d = np.diff(points, axis=0)
        return np.sum(np.linalg.norm(d, axis=1))

    # num_eval_points = target_length
    # 随机生成控制点
    left, right = 0.05, 0.95 #要给支架两边都留一点空隙，例如支架的典型半径是36个体素，那么至少要留18个体素，也就是差不多0.15的占比
    # left_2, right_2 = 0.1, 0.9 # y 和 z轴的范围还是得稍微小一点
    left_2, right_2 = 0.3, 0.7 # y 和 z轴的范围还是得稍微小一点
    control_points = []
    z_coords = np.linspace(voxel_size[2] * left, voxel_size[2] * right, num_control_points)
    # print(voxel_size[0] * left_2, voxel_size[0] * right_2, voxel_size[1] * left_2, voxel_size[1] * right_2)

    random_num = np.random.randint(0,10000)
    if(random_num%2==0):
        # 这里的点大概率会集中在体素中心
        for i in range(num_control_points):
            z_coord = z_coords[i]
            while(1):
                y_coord = np.random.uniform(voxel_size[1] * left_2, voxel_size[1] * right_2)
                x_coord = np.random.uniform(voxel_size[0] * left_2, voxel_size[0] * right_2)
                if(y_coord>=voxel_size[1] * left_2 and y_coord<voxel_size[1] * right_2 and x_coord>=voxel_size[0] * left_2 and x_coord<voxel_size[0] * right_2):
                    break
            control_points.append([x_coord, y_coord, z_coord])

    else:
        ## 这里的点应该都尽量集中于两侧
        for i in range(num_control_points):
            z_coord = z_coords[i]
            if(i%2==0):
                left,right = 0.2,0.4
                y_coord = np.random.uniform(voxel_size[1] * left, voxel_size[1] * right)
                x_coord = np.random.uniform(voxel_size[0] * left, voxel_size[0] * right)
                # while(1):
                #     y_coord = np.random.uniform(voxel_size[1] * left, voxel_size[1] * right_1)
                #     x_coord = np.random.uniform(voxel_size[0] * left_1, voxel_size[0] * right_1)
                #     if(y_coord>=voxel_size[1] * left_1 and y_coord<voxel_size[1] * right_2 and x_coord>=voxel_size[0] * left_2 and x_coord<voxel_size[0] * right_2):
                #         break
                control_points.append([x_coord, y_coord, z_coord])
            else:
                left,right = 0.6,0.8
                y_coord = np.random.uniform(voxel_size[1] * left, voxel_size[1] * right)
                x_coord = np.random.uniform(voxel_size[0] * left, voxel_size[0] * right)
                # while(1):
                #     y_coord = np.random.uniform(voxel_size[1] * left, voxel_size[1] * right_1)
                #     x_coord = np.random.uniform(voxel_size[0] * left_1, voxel_size[0] * right_1)
                #     if(y_coord>=voxel_size[1] * left_1 and y_coord<voxel_size[1] * right_2 and x_coord>=voxel_size[0] * left_2 and x_coord<voxel_size[0] * right_2):
                #         break
                control_points.append([x_coord, y_coord, z_coord])
    # print(control_points)
    
    # 创建 NURBS 曲线
    curve = NURBS.Curve()
    degree = 3 if num_control_points >= 4 else num_control_points - 1
    curve.degree = degree
    curve.ctrlpts = control_points
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.sample_size = num_eval_points
    curve.evaluate()
    
    ### 假如缩放，就有可能超出volumn范围，因此不缩放
    # 计算当前长度
    pts = np.array(curve.evalpts)
    length = curve_length(pts)

    # 缩放
    scale = target_length / length
    if(scale<1): #只有在需要缩小时候才缩放，不然要放大的话就很有可能超出范围
        ctrl_scaled = np.array(control_points)  * scale
        curve.ctrlpts = ctrl_scaled.tolist()
        curve.evaluate()
    

    point_cloud = np.array(curve.evalpts)
    # print(curve_length(pts_scaled))

    ## 移动到体素空间中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud,voxel_size=(128,128,256))

    # 绕z轴旋转是必须的
    random_angle = np.random.rand() * 180
    point_cloud = tilt_point_cloud(point_cloud,axis= 2, angle_deg = random_angle,clip=False) 

    ## 绕xy轴旋转是可选的
    rand_num = random.randint(0,1)
    if(rand_num==1): #绕y轴旋转
        random_angle = np.random.rand() * 60
        point_cloud = tilt_point_cloud(point_cloud,axis= 1, angle_deg = random_angle,clip=False) #绕y轴旋转
    else: #绕x轴旋转
        random_angle = np.random.rand() * 60
        point_cloud = tilt_point_cloud(point_cloud,axis= 0, angle_deg = random_angle,clip=False) #绕y轴旋转

    ## 再次移动到体素空间的中心
    point_cloud = center_point_cloud_to_voxel_space(point_cloud,voxel_size=(128,128,256))

    ## 将曲线完全限制在安全范围以内。
    max_x = voxel_size[0] - clip_interval
    max_y = voxel_size[1] - clip_interval
    point_cloud[:, 0] = np.clip(point_cloud[:, 0], clip_interval, max_x)
    point_cloud[:, 1] = np.clip(point_cloud[:, 1], clip_interval, max_y)
    
    return point_cloud


if __name__=='__main__':
    target_length = 100
    curve = generate_constrained_spiral_curve_iterative(target_length)
    convert_point_cloud_to_ply(curve, "d:/desktop/spiral_curve.ply")
