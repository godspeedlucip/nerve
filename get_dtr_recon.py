## 原始的trun_data的代码，和gen_truncate_data相比，我觉得与该是一样的，但是具体的也没看
## 既有recon也有truncate



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

from utils.file_rw import save_sparse_projections
from utils.general import set_geo

# def truncate_image_single_fast(image, image_mask):
#     """
#     高效版本：
#     - 不对 image 做 get_fdata
#     - 只对 mask 做一次 get_fdata
#     - 先裁剪，再 mask
#     """

#     # 1. 只解码 mask（必须）
#     mask_data = image_mask.get_fdata() > 0

#     # 2. 快速算 bounding box（不生成坐标表）
#     coords = np.where(mask_data)
#     if coords[0].size == 0:
#         print('Mask is empty')
#         return None ,None

#     start = np.array([c.min() for c in coords])
#     end   = np.array([c.max() + 1 for c in coords])  # +1 for slicing

#     # 3. 用 slicer 裁剪 image 和 mask（不触发整图加载）
#     image_roi = image.slicer[
#         start[0]:end[0],
#         start[1]:end[1],
#         start[2]:end[2],
#         ...
#     ]

#     mask_roi = image_mask.slicer[
#         start[0]:end[0],
#         start[1]:end[1],
#         start[2]:end[2],
#         ...
#     ]

#     # 4. 现在只在 ROI 内做 where（数据量小很多）
#     img_data_roi  = image_roi.get_fdata()
#     mask_data_roi = mask_roi.get_fdata() > 0

#     fill_value = img_data_roi.min()
#     masked_roi = np.where(mask_data_roi, img_data_roi, fill_value)

#     roi_nib = nib.Nifti1Image(
#         dataobj=img_data_roi,
#         affine=image_roi.affine,
#         header=image_roi.header
#     )

#     roi_masked_nib = nib.Nifti1Image(
#         dataobj=masked_roi,
#         affine=image_roi.affine,
#         header=image_roi.header
#     )

#     return roi_nib, roi_masked_nib

import scipy.sparse as sp
def load_3d_from_z_indices(sparse_dir, indices, base_name):
    """
    sparse_dir: 存 proj_000.npz, proj_001.npz ... 的目录
    indices:    list[int]
    H, W:       图像尺寸
    """
    # out = np.zeros((len(indices), H, W), dtype=np.float32)
    out = []
    for i, z in enumerate(indices):
        sp_img = sp.load_npz(os.path.join(sparse_dir, base_name + '-' +f"{z:03d}.npz"))
        out.append(sp_img.toarray())
    return np.stack(out, axis=0)

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


## 稀疏采样重建之后直接截取，不然会消耗很多存储空间
def sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter = False, already_has = None, split_json=None):
    # proj_path: 每个image-sample的投影
    # gt_path: 真实的三维标签
    # gt_mask_path: image-sample对应的vol_mask
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

    # already_has,_ = check_image(vol_train_save_path)

    # to_process = set(['ICBM_010_sample_00', 'IXI_HH_102_sample_00', 'ICBM_018_sample_00'])

    for index in tqdm(range(len(gt_mask_list))):
        i = gt_mask_list[index]

        # if(i.split('.nii.gz')[0] not in to_process):
        #     continue

        if(already_has is not None and i.split('.nii.gz')[0] in already_has.keys()):
            print( f'already processed: {i}' )
            continue
        if(split_json is not None and i.split('_sample_')[0] not in split_json.keys()):
            print(f"{i} not in to be process, continue")
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
        x_dim_max, y_dim_max, z_dim_max = vol_gt.shape[0], vol_gt.shape[1], vol_gt.shape[2]
        [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in  vol_gt_nib.header['pixdim'][1:4]]
         
        ranges = int(180 / tar_range)
        ranges_to_chose = ranges
        if(ranges > 10):
            ranges_to_chose = 10 #最多采集10个吧，太多了也不行

        ## geo只设置一次
        u_pixel_size, v_pixel_size = 0.308, 0.308
        u_detector, v_detector = 1500, 1500
        geo = set_geo(nVoxel=np.array([x_dim_max, y_dim_max, z_dim_max]), 
                dVoxel=np.array([x_voxel_size, y_voxel_size, z_voxel_size]),
                nDetector=np.array([u_detector, v_detector]),
                dDetector=np.array([u_pixel_size, v_pixel_size]))

        for j in random.sample(range(ranges), k=ranges_to_chose):
            index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.int16))
            angles = np.array([np.deg2rad(i) for i in index]) #angle根据index来
            # proj_part = proj[index] # 假如index很小，只有4，那么在写入并用ImageJ读取时就会报错，但是当index的长度为10时，就不会报错了。所以这里不用过度担心
            proj_name = i.split('.nii.gz')[0]
            proj_part = load_3d_from_z_indices(sparse_dir=join(proj_path, proj_name), base_name=proj_name.replace('_','-'),  indices=index) #直接取目标数据

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

def check_image(dirs):
    info_dict = {}
    for i in os.listdir(dirs):
        parts = i.rsplit('_', 1)
        sample_name = parts[0] # ICBM_010_sample_00
        # part2 = parts[1] # 0-60.nii.gz
        if(info_dict.get(sample_name) is None):
            info_dict[sample_name] = [i]
        else:
            info_dict[sample_name].append(i)
    info_dict_cp = copy.copy(info_dict)
    not_dict = dict()
    for i in info_dict:
        if(len(list(info_dict[i]))!=3):
            info_dict_cp.pop(i)
            not_dict[i] = info_dict[i]
    return info_dict_cp, not_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int, default=60)
    parser.add_argument("--tar_num", type=int, default=3)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_path", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--data_name", type=str, default='costa_new')
    parser.add_argument("--data_type", type=str, default='vessel')
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--use_split", type=int, default=1)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()

    tar_range = args.tar_range
    tar_num = args.tar_num
    radius = args.radius
    resulotion = args.resolution
    data_path = args.data_path
    data_name = args.data_name
    data_type = args.data_type
    split = args.split
    use_split = args.use_split
    to_train = args.train
    to_test = args.test

    # 之后生成测试数据
    print('gen train data: ')
    if(to_train):
        mode='train'
        proj_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
        gt_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt')
        gt_mask_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_dir = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/recon/{data_type}/radius-{radius}/{mode}_data')
        already_has = None

        if(use_split):
            split_file = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
            with open(split_file, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter=False, already_has = already_has, split_json=split_json)

    # 之后生成测试数据
    print('gen test data: ')
    if(to_test):
        mode='test'
        proj_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
        gt_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt')
        gt_mask_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_dir = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/recon/{data_type}/radius-{radius}/{mode}_data')
        already_has = None
        
        if(use_split):
            split_file = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
            with open(split_file, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter=False, already_has = already_has, split_json=split_json)




#### 测试代码
    # # 之后生成测试数据
    # print('gen train data: ')
    # mode='train'
    # proj_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
    # gt_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt')
    # gt_mask_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
    # save_dir = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/recon/{data_type}/radius-{radius}/{mode}_data')
    # already_has = None

    # if(use_split):
    #     split_file = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
    #     with open(split_file, "r", encoding="utf-8") as f:
    #         split_json = json.load(f)
    # else:
    #     split_json = None
    # sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter=False, already_has = already_has, split_json=split_json)

    # # 之后生成测试数据
    # print('gen test data: ')
    # mode='test'
    # proj_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
    # gt_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt')
    # gt_mask_path = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
    # save_dir = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/recon/{data_type}/radius-{radius}/{mode}_data')
    # already_has = None
    
    # if(use_split):
    #     split_file = join(data_path, f'{data_name}/voxel_{resulotion}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
    #     with open(split_file, "r", encoding="utf-8") as f:
    #         split_json = json.load(f)
    # else:
    #     split_file = None
    # sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, filter=False, already_has = already_has, split_json=split_json)
