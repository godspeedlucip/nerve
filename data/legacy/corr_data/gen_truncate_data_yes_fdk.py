## 为了获取fdk的重建对比图

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
from utils.file_rw import load_3d_from_z_indices
from utils.general import set_geo, voxelize


def truncate_image_single_fast(image, start, end, mask_roi):
    """
    高效版本：
    - 不对 image 做 get_fdata
    - 只对 mask 做一次 get_fdata
    - 先裁剪，再 mask
    """

    # 3. 用 slicer 裁剪 image 和 mask（不触发整图加载）
    image_roi = image.slicer[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
        ...
    ]

    # 4. 现在只在 ROI 内做 where（数据量小很多）
    img_data_roi  = image_roi.get_fdata()
    mask_data_roi = mask_roi.get_fdata() > 0

    fill_value = img_data_roi.min()
    masked_roi = np.where(mask_data_roi, img_data_roi, fill_value)

    roi_nib = nib.Nifti1Image(
        dataobj=img_data_roi,
        affine=image_roi.affine,
        header=image_roi.header
    )

    roi_masked_nib = nib.Nifti1Image(
        dataobj=masked_roi,
        affine=image_roi.affine,
        header=image_roi.header
    )

    return roi_nib, roi_masked_nib

def compute_mask_bbox(mask_nib):
    ## 理论上是不会越界的
    """
    只计算一次 ROI bbox（基于 mask>0），返回 (x0,x1,y0,y1,z0,z1)，注意 x1/y1/z1 为“开区间”。
    """
    # 不用 get_fdata()，避免 float64 和整幅解码
    mask = np.asanyarray(mask_nib.dataobj)  # shape: (X,Y,Z)
    # 找到非零体素的最小/最大索引
    coords = np.array(np.nonzero(mask > 0))
    if coords.size == 0:
        # 没有 ROI，直接返回整幅
        sx, sy, sz = mask.shape
        return (0, sx, 0, sy, 0, sz)

    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)

    x0, y0, z0 = mins.tolist()
    # x1, y1, z1 = (maxs + 1).tolist()  # 开区间 # 用这个就会出现 2 * radius + 1的情况
    x1, y1, z1 = maxs.tolist()  # 开区间
    start = np.array((x0, y0, z0))
    end   = np.array((x1, y1, z1))  # +1 会导致体素数比一般的多1个, 不+1就正好
    return start, end

def check_trun_data_complexity(trun_data_save_dir):
    all_dict = []
    not_finish = []
    for i in os.listdir(trun_data_save_dir):
        single_dir = join(trun_data_save_dir, i)
        all_num = dict()
        for j in os.listdir(single_dir):
            idx = j.split("_sample_")[0] + "_sample_" + j.split("_sample_")[1].split("_")[0]
            if(all_num.get(idx) is None):
                all_num[idx] = {'num':1, 'name': join(i,j)}
            else:
                all_num[idx]['num'] += 1
        all_dict.append(all_num)
    
    for i in all_dict:
        for j in i:
            if(i[j]['num']!=3):
                # print(i[j]['name'])
                # print(j)
                not_finish.append(j)
    return list(set(all_dict[0].keys()) - set(not_finish))


## 稀疏采样重建之后直接截取，不然会消耗很多存储空间
def sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter = False, already_has = None):
    if(filter):
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkYesFilter")
    else:
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkNoFilter")
    
    vol_train_save_path_no_masked = join(save_path,'vol_train_no_masked')
    vol_gt_save_path_no_masked = join(save_path,'vol_gt_no_masked')
    vol_train_save_path = join(save_path,'vol_train')
    vol_gt_save_path = join(save_path,'vol_gt')
    # print(vol_train_save_path, vol_gt_save_path, vol_train_save_path_no_masked, vol_gt_save_path_no_masked)
    os.makedirs(vol_train_save_path, exist_ok=True)
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(vol_train_save_path_no_masked, exist_ok=True)
    os.makedirs(vol_gt_save_path_no_masked, exist_ok=True)

    # 180度范围内，尽量多取几例数据来训练
    # proj_list = sorted(os.listdir(proj_path))
    gt_list = sorted(os.listdir(gt_path))
    gt_mask_list = sorted(os.listdir(gt_mask_path))

    for index in tqdm(range(len(gt_mask_list))):
        i = gt_mask_list[index]
        if(already_has is not None and i in already_has):
            print( f'already processed: {i}' )
            continue
        
        # proj  = np.load(join(proj_path,proj_list[index]))['data'] # 投影是npz格式. 这里的Proj默认就是 z, x, y的格式
        vol_gt_mask_nib = nib.load(join(gt_mask_path, gt_list[index]))
        vol_gt_nib = nib.load(join(gt_path, gt_list[index])) # 图像是nii.gz格式
        vol_gt = np.transpose(vol_gt_nib.get_fdata(), [2, 0, 1])

        ## 提前计算截取的边界
        start, end = compute_mask_bbox(vol_gt_mask_nib)

        ## mask先截
        mask_roi = vol_gt_mask_nib.slicer[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2],
            ...
        ]

        vol_gt_truncated_no_masked, vol_gt_truncated = truncate_image_single_fast(vol_gt_nib, start, end, mask_roi)

        ## 每个文件对应的投影参数都不是一样的
        z_dim_max, x_dim_max, y_dim_max = vol_gt.shape[0], vol_gt.shape[1], vol_gt.shape[2]
        [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in  vol_gt_nib.header['pixdim'][1:4]]
         
        ranges = int(180 / tar_range)
        ranges_to_chose = ranges
        if(ranges > 10):
            ranges_to_chose = 10 #最多采集10个吧，太多了也不行

        ## geo只设置一次
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                    dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                    nDetector=np.array([1024*2, 1024*2]),
                    dDetector=np.array([x_voxel_size, y_voxel_size]))

        for j in random.sample(range(ranges), k=ranges_to_chose):
            index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.int16))
            angles = np.array([np.deg2rad(i) for i in index]) #angle根据index来
            # proj_part = proj[index] # 假如index很小，只有4，那么在写入并用ImageJ读取时就会报错，但是当index的长度为10时，就不会报错了。所以这里不用过度担心
            proj_part = load_3d_from_z_indices(sparse_dir=join(proj_path, i.split('.nii.gz')[0]), indices=index) #直接取目标数据
            if(filter):
                imgFDK = algs.fdk(proj_part, geo, angles) #有限角的重建结果
            else:
                imgFDK = algs.fdk_no_filter(proj_part, geo, angles) #fdk不用滤波重建
                
            imgFDK_nib  = nib.Nifti1Image(
                dataobj=np.transpose(imgFDK, [1, 2, 0]), #nii文件中图像的维度顺序是x, y, z
                affine=vol_gt_nib.affine,
                header=vol_gt_nib.header
            )

            vol_train_truncated_no_masked, vol_train_truncated = truncate_image_single_fast(imgFDK_nib, start, end, mask_roi)

            file_name = f'{i.split(".")[0]}_{str(int(tar_range*j))}-{str(int(tar_range*(j+1)))}.nii.gz'
            nib.save(vol_train_truncated_no_masked, join(vol_train_save_path_no_masked, file_name))
            nib.save(vol_gt_truncated_no_masked, join(vol_gt_save_path_no_masked, file_name))
            nib.save(vol_train_truncated, join(vol_train_save_path, file_name))
            nib.save(vol_gt_truncated, join(vol_gt_save_path, file_name))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int, default=60)
    parser.add_argument("--tar_num", type=int, required=True)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    parser.add_argument("--data_type", type=str, required=True) #stent vessel
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/interior/data/corr_data/data')
    args = parser.parse_args()

    tar_range = args.tar_range
    tar_num = args.tar_num
    radius = args.radius
    resolution = args.resolution
    data_dir = args.data_dir
    data_name = args.data_name
    data_type = args.data_type
    to_train = args.train
    to_test = args.test
    assert data_type in set(['stent', 'vessel'])
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    
    data_dir = join(data_dir, data_name)

    print(f'tar_range: {tar_range}\n',
          f'tar_num: {tar_num}\n',
          f'data_dir: {data_dir}\n',
          f'data_type: {data_type}\n',
          f'radius: {radius}\n',
          f'resolution: {resolution}\n',
          f'type: {data_type}')


    
    if(to_train):
        print('生成训练数据')
        mode='train'
        train_proj_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/proj_gt')
        train_gt_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/vol_gt')
        train_gt_mask_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/vol_gt_mask')
        train_save_dir = join(data_dir, f'voxel_{resolution}/trun_data/radius-{radius}/corr_{data_type}/{mode}_data')
        already_has = None
        sparse_recon(proj_path=train_proj_path, gt_path=train_gt_path, gt_mask_path=train_gt_mask_path, save_dir=train_save_dir, tar_range=tar_range, tar_num=tar_num, filter=True, already_has = already_has)

    if(to_test):
        print('生成测试数据')
        mode='test'
        train_proj_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/proj_gt')
        train_gt_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/vol_gt')
        train_gt_mask_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/corr_{data_type}/{mode}_data/vol_gt_mask')
        train_save_dir = join(data_dir, f'voxel_{resolution}/trun_data_yes_fdk/radius-{radius}/corr_{data_type}/{mode}_data')
        already_has = None
        sparse_recon(proj_path=train_proj_path, gt_path=train_gt_path, gt_mask_path=train_gt_mask_path, save_dir=train_save_dir, tar_range=tar_range, tar_num=tar_num, filter=True, already_has = already_has)



