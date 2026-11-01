import numpy as np
import plyfile, copy, pickle, os, random, json, argparse
import pandas as pd
import tifffile as tiff
from scipy import io
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk
from scipy.ndimage import convolve, gaussian_filter
import scipy.sparse as sp
import tigre
import tigre.algorithms as algs
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import zoom
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import uniform_filter
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

from utils.file_rw import read_pkl, save_sparse_projections
from utils.general import voxelize, set_geo, check_data
from utils.centerline import remove_small_components

# def sample_on_original_points_pca(points: np.ndarray, num_samples: int = 10):
#     """
#     通过 PCA 找到最长轴，将点云排序后，沿着该轴均匀选择原始点云上的点。
    
#     参数:
#         points: np.ndarray (N, 3) 原始点云坐标
#         num_samples: int，采样数量
#     返回:
#         samples: np.ndarray (num_samples, 3) 选自原始点云的坐标
#     """
#     if len(points) < num_samples:
#         return points # 点太少，返回全部

#     # 1. PCA 分解 (与原代码相同)
#     centered = points - points.mean(axis=0)
#     cov = np.cov(centered.T)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     order = np.argsort(eigvals)[::-1]
#     eigvecs = eigvecs[:, order]
#     main_dir = eigvecs[:, 0]

#     # 2. 投影到主轴，并根据投影值排序
#     projections = centered @ main_dir
#     # 获取排序后的索引：沿着主轴从小到大排列
#     sorted_indices = np.argsort(projections)

#     # 3. 均匀采样索引
#     total_points = len(points)
    
#     # 确定采样间隔
#     interval = total_points / num_samples
    
#     # 确定一个随机起始偏移量 (确保随机性)
#     start_offset = np.random.uniform(0, interval)

#     # 生成 num_samples 个均匀采样的浮点索引
#     floating_indices = np.arange(num_samples) * interval + start_offset
    
#     # 将浮点索引转换为整数索引 (并取模确保不越界)
#     sample_indices = np.floor(floating_indices).astype(np.int32) % total_points

#     # 4. 从排序后的索引中取出采样的点
#     # 选中排序后列表中的这些点
#     selected_indices_in_original = sorted_indices[sample_indices]
    
#     # 返回原始坐标
#     samples_on_original = points[selected_indices_in_original]
    
#     return samples_on_original

def get_circle_roi_fast(image_shape, roi_center, roi_radius):
    cx, cy, cz = roi_center
    r = int(np.ceil(roi_radius))
    Nx, Ny, Nz = image_shape

    # 1. 确定 Bounding Box 的范围（防止越界）
    extra_bounding = int(r * 0.25)
    x_min, x_max = max(0, cx - (r+extra_bounding)), min(Nx, cx + (r+extra_bounding) + 1)
    y_min, y_max = max(0, cy - (r+extra_bounding)), min(Ny, cy + (r+extra_bounding) + 1)
    z_min, z_max = max(0, cz - (r+extra_bounding)), min(Nz, cz + (r+extra_bounding) + 1)

    # 2. 仅在局部区域生成网格
    x, y, z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]

    # 3. 计算距离平方（利用广播机制，ogrid 比 mgrid 更省内存）
    dist_sq = (z - cz)**2 + (x - cx)**2 + (y - cy)**2
    
    # 4. 筛选并恢复全局坐标
    mask = dist_sq <= roi_radius**2
    local_coords = np.where(mask)
    
    global_x = local_coords[0] + x_min
    global_y = local_coords[1] + y_min
    global_z = local_coords[2] + z_min

    # return np.vstack([global_x, global_y, global_z]), np.array([x_min, y_min, z_min], [x_max, y_max, z_max])
    return np.vstack([global_x, global_y, global_z])

def get_roi_proj(src_proj, geo, angles, valid_mesh, image_sizes):
    """
    valid_mesh: (3,num_of_points). roi区域的点云形式
    src_proj: 原始图像的投影
    image_sizes: 原始图像的shape
    """
    ## 创建一个只有ROI区域有值的体膜
    roi_image_mask = np.zeros(image_sizes, np.float32)
    x_coords, y_coords, z_coords = valid_mesh[0], valid_mesh[1], valid_mesh[2]
    roi_image_mask[x_coords, y_coords, z_coords] = 1

    ## 对roi区域投影
    roi_mask = tigre.Ax(np.transpose(roi_image_mask, [2, 0, 1]), geo, angles)
    roi_mask[roi_mask>0.01] = 1 #将其转换为mask
    roi_mask[roi_mask<=0.01] = 0

    #逐元素相乘提取数据
    roi_proj = src_proj * roi_mask
    return roi_proj, roi_image_mask


def get_roi(mask, sample_point, src_proj, radius=64, angles = np.linspace(0,2*np.pi,360), geo = None):
    """
    mask: 图像的mask
    sample_point: 一个中心点
    para_path: 图像的参数的文件
    radius: 采样圆的半径
    """
    roi_center = np.array([sample_point[0], sample_point[1], sample_point[2]])
    point_mesh = get_circle_roi_fast(mask.shape, roi_center, radius) # (3, N)的点云形式 #真值
    roi_proj, roi_gt_mask = get_roi_proj(src_proj, geo, angles, point_mesh, image_sizes=mask.shape) #这个函数是通过点的坐标去构建三维体素，因此就算传入的mask是(x, y, z)的shape也没关系
    fill_value = np.min(mask)
    roi_gt = np.where(roi_gt_mask, mask, fill_value)
    return roi_proj, roi_gt, roi_gt_mask

def get_random_points(final_components, total_samples=10):
    if not final_components:
        return np.array([])
    selected_points = []
    num_lines = len(final_components)
    # 循环抽取，直到凑够 10 个点
    for _ in range(total_samples):
        # 1. 随机选一条线
        target_line = random.choice(final_components)
        # 2. 从这条线里随机选一个点
        idx = np.random.randint(0, len(target_line))
        selected_points.append(target_line[idx])

    return np.array(selected_points)

def save_ply_with_o3d(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

# 下面这个work的思路是直接得到真实图像和训练图像，但是应该是先得到完整的投影和图像，再选择角度去反投影。最后一步才是裁剪，
def main(data_dir, save_dir = './', number_roi_in_one_sample = 10, radius=64, already_has = None, resolution = 0.2, mode='train'):
    # 设置保存路径
    point_cloud_save_path = join(save_dir,'point_cloud')
    os.makedirs(point_cloud_save_path, exist_ok=True)
    print(point_cloud_save_path)

    vol_gt_save_path = join(save_dir,'vol_gt')
    proj_gt_save_path = join(save_dir,'proj_gt')
    vol_gt_mask_save_path = join(save_dir,'vol_gt_mask')
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(proj_gt_save_path, exist_ok=True)
    os.makedirs(vol_gt_mask_save_path, exist_ok=True)

    ## 一些提前就创建完成的文件
    resample_dir = join(data_dir, 'resample')
    distance_dir = join(data_dir, 'distance_map')
    skeleton_dir = join(data_dir, 'skeleton')

    ## 对每个mask，都在主动脉区域选取10个点，并截取以这个点为圆心的圆形区域作为训练数据

    for i in tqdm(os.listdir(resample_dir)):
        if(already_has is not None and i.split('.nii.gz')[0] in already_has):
            print('already: ', i)
            continue
        
        ## 加载必要的数据
        nib_mask = nib.load(join(resample_dir,i)) #(w,h,d). 假设w是x的话，那么h就是y。那么确实是 (x,y,z)排列的
        distance_mask = np.load(join(distance_dir, i.split('.nii.gz')[0]+'.npz'))['data'] 
        skeleton_mask = nib.load(join(skeleton_dir,i)).get_fdata()
        mask = nib_mask.get_fdata()
        # mask = mask.astype(np.float32)

        # 设置几何参数
        angles = np.linspace(0, 2*np.pi, 360)
        radius = radius
        x_dim_max, y_dim_max, z_dim_max = mask.shape[0], mask.shape[1], mask.shape[2] #voxel size
        x_voxel_size, y_voxel_size, z_voxel_size = [round(i, 4) for i in  nib_mask.header['pixdim'][1:4]] #voxel space
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array([1024*2, 1024*2]), #投影要大一点，因为重采样后的数据会很大
            dDetector=np.array([resolution, resolution]))
        src_proj = tigre.Ax(np.transpose(mask.astype(np.float32), [2, 0, 1]), geo, angles) #原始图像的投影

        ## 来选择合适的点
        min_radius = 5
        max_radius = 11
        skeleton_point_cloud = np.vstack([np.where(skeleton_mask > 0)]).T
        z_coords, x_coords, y_coords = skeleton_point_cloud[:, 0], skeleton_point_cloud[:, 1], skeleton_point_cloud[:, 2]
        local_radii = distance_mask[z_coords, x_coords, y_coords]
        thick_indices = (local_radii >= min_radius) & (local_radii < max_radius)
        ok_point_cloud = skeleton_point_cloud[thick_indices,:]
        points_filter, components = remove_small_components(ok_point_cloud, k = 5, max_connect_dist = 2, min_component_size=35)
        sample_points = get_random_points(components, number_roi_in_one_sample)

        points_info = dict()
        points_info['components'] = components
        points_info['sample_points'] = sample_points
        with open(join(point_cloud_save_path, f"{i.split('.nii.gz')[0]}.pickle"), 'wb') as f:
            pickle.dump(points_info, f)

        # save_ply_with_o3d(join(save_dir, 'points'))
        # 首先将roi区域内的mask内容提取出来，作为图像的真值
        for sample_index, sample_point in enumerate(sample_points):
            # 获得该中心点对应的roi
            roi_proj, roi_gt, roi_image_mask = get_roi(mask=mask, sample_point=sample_point, src_proj=src_proj, angles=angles, radius=radius, geo=geo) #与原始投影、原始图像的大小是一样的，但是只有ROI的区域有值，其他地方都是0

            # 把roi_gt从(z, x, y)变为(x, y, z)
            # roi_gt = np.transpose(roi_gt, [1, 2, 0])
            roi_gt_nib  = nib.Nifti1Image(
                dataobj=roi_gt,
                affine=nib_mask.affine,
                header=nib_mask.header
            )
            # roi_image_mask = np.transpose(roi_image_mask, [1, 2, 0])
            roi_gt_mask_nib  = nib.Nifti1Image(
                dataobj=roi_image_mask,
                affine=nib_mask.affine,
                header=nib_mask.header
            )

            save_name = f"{i.split('.')[0]}_sample_{str(sample_index).zfill(2)}"
            nib.save(roi_gt_nib, join(vol_gt_save_path, save_name+'.nii.gz'))
            nib.save(roi_gt_mask_nib, join(vol_gt_mask_save_path, save_name+'.nii.gz'))
            save_sparse_projections(proj=roi_proj, out_dir=join(proj_gt_save_path, save_name))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--data_path", type=str, default='/data1/liuyang/stent/interior/data/topbrain/helper')
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    data_path = args.data_path
    save_path = args.save_path

    print(f'radius: {radius}\n',
          f'data_path: {data_path}',
          f"save_path: ",{save_path})


    mode='train'
    print('处理训练数据')
    final_data_path = join(data_path, f'{mode}_data/voxel_{resolution}')
    final_save_path = join(save_path, f"voxel_{resolution}/src_data/radius-{radius}/{mode}_data")
    main(data_dir = final_data_path, save_dir = final_save_path, radius=radius, already_has = None, resolution=resolution)

    mode='test'
    print('处理训练数据')
    final_data_path = join(data_path, f'{mode}_data/voxel_{resolution}')
    final_save_path = join(save_path, f"voxel_{resolution}/src_data/radius-{radius}/{mode}_data")
    main(data_dir = final_data_path, save_dir = final_save_path, radius=radius, already_has = None, resolution=resolution)

