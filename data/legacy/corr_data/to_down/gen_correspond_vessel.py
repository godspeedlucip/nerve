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

from gen_stent.utils.base_utils import stent_simu, expand_and_smooth_voxel_mask, voxelize_1, voxelize_2, center_point_cloud_to_voxel_space, set_geo, flip_point_cloud, get_min_max, visualize_braid_stent
from gen_stent.utils.gen_curve import generate_constrained_spiral_curve_iterative, generate_single_sin_curve, generate_3d_sine_curve_pointcloud, generate_random_smooth_curve, create_nurbs_centerline_fixed_length, create_outward_arc_u_shape_point_cloud
from gen_stent.utils.curve_trans import transform_pointcloud_frenet_serret, transform_pointcloud_parallel, compute_frenet_serret, compute_parallel_transport_frames
from gen_stent.utils.file_rw import read_ply
from gen_stent.utils.gen_stent import w_like_Stent, ring_Stent, Stent_Ply, sin_Stent
from gen_stent.utils.get_stent_para import map_stent_to_curve
from utils.file_rw import read_pkl, save_sparse_projections

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

# def get_rectangle_roi(roi_center,roi_size): #获得长方形区域的roi坐标
#     # box_min = (roi_center - roi_size / 2.0).astype(np.uint8)
#     # box_max = (roi_center + roi_size / 2.0).astype(np.uint8)
#     box_min = (roi_center - roi_size / 2.0).astype(np.uint16)
#     box_max = (roi_center + roi_size / 2.0).astype(np.uint16)
#     x_coords, y_coords ,z_coords= np.mgrid[
#         box_min[0] : box_max[0] + 1,
#         box_min[1] : box_max[1] + 1,
#         box_min[2] : box_max[2] + 1
#     ]
#     return np.vstack([x_coords.ravel(), y_coords.ravel() ,z_coords.ravel()])

# def get_circle_roi(image_shape, roi_center, roi_radius): #获得圆形区域的roi坐标
#     Nz, Nx, Ny = image_shape
#     center_z, center_x, center_y = roi_center
    
#     z_coords, x_coords, y_coords = np.mgrid[0:Nz, 0:Nx, 0:Ny]

#     #计算每个体素到圆心在 XYZ 平面上的距离平方
#     distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2 + (z_coords - center_z)**2
#     radius_sq = roi_radius**2
#     mask = distance_sq <= radius_sq

#     mask = mask.astype(np.int8)
#     z,x,y = np.where(mask>0)
#     return np.vstack([z,x,y])

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


# def get_roi_111(mask, sample_point, src_proj, radius=64, angles = np.linspace(0,2*np.pi,360), geo = None):
#     """
#     mask: 图像的mask
#     sample_point: 一个中心点
#     para_path: 图像的参数的文件
#     radius: 采样圆的半径
#     """
#     roi_center = np.array([sample_point[0], sample_point[1], sample_point[2]])
#     point_mesh, coords = get_circle_roi_fast(mask.shape, roi_center, radius) # (3, N)的点云形式 #真值
#     roi_proj, roi_gt_mask = get_roi_proj(src_proj, geo, angles, point_mesh, image_sizes=mask.shape) #这个函数是通过点的坐标去构建三维体素，因此就算传入的mask是(x, y, z)的shape也没关系
#     # fill_value = np.min(mask)
#     # roi_gt = np.where(roi_gt_mask, mask, fill_value)
#     roi_gt_rantangle = np.zeros_like(mask, dtype=mask.dtype) #在球形ROI外面->外接正方形里面保留了值的图像
#     roi_gt_rantangle[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1], coords[0][2]:coords[1][2]] = mask[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1], coords[0][2]:coords[1][2]]
#     return roi_proj, roi_gt_rantangle, roi_gt_mask

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
    
    ## 确保索引不超出范围
    valid_mask = (Z_indices >= 0) & (Z_indices < D) & \
                 (X_indices >= 0) & (X_indices < H) & \
                 (Y_indices >= 0) & (Y_indices < W)

    valid_Z = Z_indices[valid_mask]
    valid_X = X_indices[valid_mask]
    valid_Y = Y_indices[valid_mask]
    
    valid_indices = (valid_Z, valid_X, valid_Y)
    voxel = np.zeros(voxel_size, dtype=np.float32) 
    voxel[valid_indices] = fill_value
    
    return voxel
    

# 下面这个work的思路是直接得到真实图像和训练图像，但是应该是先得到完整的投影和图像，再选择角度去反投影。最后一步才是裁剪，
def main(data_dir, save_dir = './', config_path = None, radius=60, already_has = None, resolution = (0.2, 0.2, 0.2)):

    vol_gt_save_path = join(save_dir,'vol_gt')
    proj_gt_save_path = join(save_dir,'proj_gt')
    vol_gt_mask_save_path = join(save_dir,'vol_gt_mask')
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(proj_gt_save_path, exist_ok=True)
    os.makedirs(vol_gt_mask_save_path, exist_ok=True)

    ## 一些提前就创建完成的文件
    resample_dir = join(data_dir, 'resample')

    ## 对每个mask，都在主动脉区域选取10个点，并截取以这个点为圆心的圆形区域作为训练数据
    for i in tqdm(os.listdir(resample_dir)):
        if(already_has is not None and i.split('.nii.gz')[0] in already_has):
            print('already: ', i)
            continue
        
        ## 加载必要的数据
        pkl_path = join(config_path, i.split('.nii.gz')[0] + '.pickle')
        if(os.path.exists(pkl_path)==False):
            print(f"{i} 没有对应的config")
            continue
        point_config = read_pkl(pkl_path)

        nib_mask = nib.load(join(resample_dir,i)) #(w,h,d). 假设w是x的话，那么h就是y。那么确实是 (x,y,z)排列的
        mask = nib_mask.get_fdata().astype(np.float32)
        

        # 设置几何参数
        angles = np.linspace(0, 2*np.pi, 360)
        radius = radius
        x_dim_max, y_dim_max, z_dim_max = mask.shape[0], mask.shape[1], mask.shape[2] #voxel size
        x_voxel_size, y_voxel_size, z_voxel_size = [round(i, 4) for i in  nib_mask.header['pixdim'][1:4]] #voxel space
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array([1024*2, 1024*2]), #投影要大一点，因为重采样后的数据会很大
            dDetector=np.array([resolution[0], resolution[1]]))
        src_proj = tigre.Ax(np.transpose(mask, [2, 0, 1]), geo, angles) #原始图像的投影

        for idx in point_config:
            idx_center = point_config[idx]['center']
            idx_border = point_config[idx]['border'] #这行代码实际上没有用

            idx_center = np.array(idx_center).astype(np.int16)

            # 获得该中心点对应的roi. 图像都是x y z的顺序, 而投影都是转换到z, x, y的顺序去投的
            roi_proj, roi_gt, roi_image_mask = get_roi(mask=mask, sample_point=idx_center, src_proj=src_proj, angles=angles, radius=radius, geo=geo) #与原始投影、原始图像的大小是一样的，但是只有ROI的区域有值，其他地方都是0

            roi_gt_nib  = nib.Nifti1Image(
                dataobj=roi_gt,
                affine=nib_mask.affine,
                header=nib_mask.header
            )
            roi_gt_mask_nib  = nib.Nifti1Image(
                dataobj=roi_image_mask,
                affine=nib_mask.affine,
                header=nib_mask.header
            )

            save_name = f"{i.split('.')[0]}_sample_{str(idx).zfill(2)}"
            nib.save(roi_gt_nib, join(vol_gt_save_path, save_name+'.nii.gz'))
            nib.save(roi_gt_mask_nib, join(vol_gt_mask_save_path, save_name+'.nii.gz'))
            save_sparse_projections(proj=roi_proj, out_dir=join(proj_gt_save_path, save_name))
            # np.savez_compressed(join(proj_gt_save_path, save_name+'.npz'),data=roi_proj)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_dir",type=str, default='/data2/liuyang/liuyang/stent/interior/data/costa/helper')
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    # stent_path = args.stent_path
    save_path = args.save_dir
    resulotion = args.resolution
    radius = args.radius

    ## 生成训练和测试数据
    print('生成训练数据')
    mode='train'
    final_save_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_vessel/{mode}_data')
    final_data_dir = join(data_dir, f'{mode}_data/voxel_{resulotion}')
    final_config_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data/config')
    has_proceed = ['ICBM_016','IXI_HH_137','IXI_Guys_100','IXI_Guys_084','IXI_HH_102','IXI_HH_168','IXI_HH_104','IXI_HH_214','IXI_HH_165']
    main(data_dir=final_data_dir, save_dir=final_save_dir, config_path=final_config_dir, radius=radius, already_has = set(has_proceed))

    ## 生成训练和测试数据
    print('生成测试数据')
    mode='test'
    final_save_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_vessel/{mode}_data')
    final_data_dir = join(data_dir, f'{mode}_data/voxel_{resulotion}')
    final_config_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data/config')
    main(data_dir=final_data_dir, save_dir=final_save_dir, config_path=final_config_dir, radius=radius)





    