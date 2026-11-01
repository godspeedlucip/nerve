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
import open3d as o3d
import random



## Frenet-Serret框架来变换
def compute_frenet_serret(curve_points, eps=1e-6): #计算曲线中每个点的局部坐标
    """
    计算 Frenet-Serret 坐标系向量。
    """
    T = np.gradient(curve_points, axis=1)
    T_norm = np.linalg.norm(T, axis=0)
    T_hat = T / (T_norm + eps)

    T_hat_prime = np.gradient(T_hat, axis=1)
    N_norm = np.linalg.norm(T_hat_prime, axis=0)
    N_hat = T_hat_prime / (N_norm + eps)
    
    # print(N_hat.shape, T_hat.shape)
    B_hat = np.cross(T_hat, N_hat, axis=0)
    return T_hat, N_hat, B_hat

def transform_pointcloud_frenet_serret(orig_points, curve_points, curve_N, curve_B):
    """
    将点云变换到曲线中心线上。
    orig_points: (N,3) #初始的点云
    curve_points: (3, num_z) 曲线上的点
    curve_N, curve_B: (3, num_z) 曲线上各点的法向量
    """
    num_z = curve_points.shape[1] #曲线上点的数量

    # 归一化原 cylinder z -> [0,1]
    z_orig = orig_points[:, 2]
    z_param = z_orig / z_orig.max() #把原始点云的z轴归一化

    # 插值找对应的曲线点和局部基
    new_points = []
    for i, t in enumerate(z_param): #对于初始点云的每个点的z坐标
        idx = int(t * (num_z - 1)) #计算这个点在曲线中的位置
        C = curve_points[:, idx] #由于是按照z轴变换，因此z坐标肯定就是相等的
        N = curve_N[:, idx]
        B = curve_B[:, idx]
        x, y = orig_points[i, 0], orig_points[i, 1] #原始点云的xy坐标
        P_new = C + x * N + y * B #向量运算，得到的是最终位置
        new_points.append(P_new)
    return np.array(new_points)



## 平行框架来变换
def compute_parallel_transport_frames(curve_pts, eps=1e-12):
    """
    计算沿曲线的平行运输 (parallel transport) 框架。
    输入:
      curve_pts: (M,3) 曲线上均匀采样的点
    返回:
      T: (M,3) 切向量（单位）
      N: (M,3) 法向量（单位）
      B: (M,3) 副法向量（单位）
    说明:
      使用最小角度旋转将前一个截面的 N 向量传输到当前截面，避免跳变。
    """
    def normalize(v, eps=1e-12):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norm + eps)
    M = curve_pts.shape[0]
    # 计算切向量（中心差分，边界用前后差）
    # 用 np.gradient 更稳健（会输出 (M,3)）
    T = np.gradient(curve_pts, axis=0)
    T = normalize(T, eps=eps)

    N = np.zeros_like(T)
    B = np.zeros_like(T)

    # 初始法向量 N0: 选择一个与 T0 不共线的参考向量（例如全局 z），然后取垂直分量并归一化
    T0 = T[0]
    # 选择参考向量 r 使得 r not parallel T0
    r = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T0, r)) > 0.9:
        r = np.array([1.0, 0.0, 0.0])  # 如果太接近，换另一个方向
    # N0 为 r 在垂直于 T0 平面的投影
    N0 = r - np.dot(r, T0) * T0
    N0 = N0 / (np.linalg.norm(N0) + eps)
    N[0] = N0
    B[0] = np.cross(T0, N0)
    B[0] = B[0] / (np.linalg.norm(B[0]) + eps)

    # 逐步平行运输
    for i in range(1, M):
        v_prev = T[i-1]
        v_cur  = T[i]

        # 若两个切向量几乎平行，则保持 N 不变
        dot = np.dot(v_prev, v_cur)
        dot = np.clip(dot, -1.0, 1.0)
        if np.isclose(dot, 1.0, atol=1e-8):
            # 切向量方向几乎相同，保持 N
            N[i] = N[i-1]
        else:
            # 旋转轴
            axis = np.cross(v_prev, v_cur)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-12:
                # 切向量反向或数值问题，直接保持 N（或取一个正交方向）
                N[i] = N[i-1]
            else:
                axis = axis / axis_norm
                angle = np.arccos(dot)
                # Rodrigues 旋转公式：把 N[i-1] 绕 axis 旋转 angle 得到 N[i]
                u = axis
                n_prev = N[i-1]
                # 公式: n' = n cosθ + (u × n) sinθ + u (u·n) (1-cosθ)
                n_rot = (n_prev * np.cos(angle) +
                         np.cross(u, n_prev) * np.sin(angle) +
                         u * (np.dot(u, n_prev)) * (1 - np.cos(angle)))
                # 如果 n_rot 与 v_cur 不正交（数值误差），强制正交化并归一化
                # 投影到垂直于 v_cur 的平面
                n_rot = n_rot - np.dot(n_rot, v_cur) * v_cur
                n_rot = n_rot / (np.linalg.norm(n_rot) + eps)
                N[i] = n_rot

        # 计算 B
        B_i = np.cross(T[i], N[i])
        B_i = B_i / (np.linalg.norm(B_i) + eps)
        B[i] = B_i

    return T, N, B

def transform_pointcloud_parallel(orig_points, curve_pts, N_frames, B_frames):
    """
    使用平行运输框架把原始圆柱形点云沿曲线变形。
    输入:
      orig_points: (K,3) 原始点云，假设 z 是沿长度方向且范围 [0, z_max]
      curve_pts: (M,3) 曲线采样点
      N_frames, B_frames: (M,3) 对应法向量和副法向量
    返回:
      new_points: (K,3) 变换后的点云
    实现细节:
      对每个原始点，通过 z 参数映射到曲线上（线性插值），对 C,N,B 做线性插值（在向量尺度），
      然后 P_new = C + x * N + y * B。
    """
    # 确保输入格式
    curve_pts = np.asarray(curve_pts)
    N_frames = np.asarray(N_frames)
    B_frames = np.asarray(B_frames)

    M = curve_pts.shape[0]
    K = orig_points.shape[0]

    # 参数化原始点云 z -> t in [0,1]
    z_orig = orig_points[:, 2]
    z_max = z_orig.max() if z_orig.max() > 0 else 1.0
    t_vals = z_orig / z_max
    t_vals = np.clip(t_vals, 0.0, 1.0)

    # 计算浮点索引
    idx_f = t_vals * (M - 1)
    idx0 = np.floor(idx_f).astype(int)
    idx1 = np.minimum(idx0 + 1, M - 1)
    alpha = idx_f - idx0  # in [0,1)

    new_points = np.zeros_like(orig_points, dtype=float)
    x_orig = orig_points[:, 0]
    y_orig = orig_points[:, 1]

    # 对每个点线性插值 C,N,B
    C0 = curve_pts[idx0]
    C1 = curve_pts[idx1]
    N0 = N_frames[idx0]
    N1 = N_frames[idx1]
    B0 = B_frames[idx0]
    B1 = B_frames[idx1]

    alpha = alpha[:, None]  # shape (K,1) 便于广播
    C_interp = (1 - alpha) * C0 + alpha * C1
    N_interp = (1 - alpha) * N0 + alpha * N1
    B_interp = (1 - alpha) * B0 + alpha * B1

    # 归一化插值后的 N,B（线性插值后可能不正交或未归一化）
    N_interp = N_interp - np.sum(N_interp * ( (1 - alpha) * (np.zeros_like(N0) ) ), axis=1, keepdims=True) # no-op to satisfy shape
    # 简单归一化
    N_interp = N_interp / (np.linalg.norm(N_interp, axis=1, keepdims=True) + 1e-12)
    B_interp = B_interp - np.sum(B_interp * ( (1 - alpha) * (np.zeros_like(B0) ) ), axis=1, keepdims=True) # no-op
    B_interp = B_interp / (np.linalg.norm(B_interp, axis=1, keepdims=True) + 1e-12)

    # 最后构造新点
    new_points = C_interp + x_orig[:, None] * N_interp + y_orig[:, None] * B_interp

    return new_points

