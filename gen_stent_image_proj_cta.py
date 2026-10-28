### 把支架放到处理之后的MRI图像中



import numpy as np
import plyfile, copy, pickle, os, random, json, argparse
import pandas as pd
import tifffile as tiff
from scipy import io
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
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
import scipy.sparse as sp

from utils.file_rw import load_3d_from_z_indices, write_sp, read_sp
from utils.general import set_geo, voxelize, coarse_mri_to_ct, hu_to_mu
from utils.marker import compose_stent_and_marker

def load_roi_mask_proj(proj_dir):
    proj = []
    for i in sorted(os.listdir(proj_dir)):
        proj.append(read_sp(join(proj_dir, i)))
    return np.stack(proj, axis=0)

def gen_roi_proj(stent_path, image_path, vol_gt_mask_path, save_dir, split_json,
                 marker_path=None, marker_mu_ratio=2.0):
    ## 创建输出路径
    # stent_within_image_save_dir = join(save_dir, 'image')
    # roi_proj_save_dir = join(save_dir, 'roi_proj')
    # stent_proj_save_dir = join(save_dir, 'stent_proj')

    # roi_proj_save_dir = join(save_dir, 'roi_proj')
    stent_within_image_save_dir = join(save_dir, 'image')
    stent_within_label_save_dir = join(save_dir, 'label')
    cir_all_image_proj_save_dir = join(save_dir, 'image_proj', 'cir_all')
    rec_image_proj_save_dir = join(save_dir, 'image_proj', 'rec')
    rec_all_image_proj_save_dir = join(save_dir, 'image_proj', 'rec_all')
    cir_image_proj_save_dir = join(save_dir, 'image_proj', 'cir')
    cir_all_label_proj_save_dir = join(save_dir, 'label_proj', 'cir_all')
    rec_label_proj_save_dir = join(save_dir, 'label_proj', 'rec')
    rec_all_label_proj_save_dir = join(save_dir, 'label_proj', 'rec_all')
    cir_label_proj_save_dir = join(save_dir, 'label_proj', 'cir')
    # os.makedirs(roi_proj_save_dir, exist_ok=True)
    os.makedirs(stent_within_image_save_dir, exist_ok=True)
    os.makedirs(stent_within_label_save_dir, exist_ok=True)
    
    ## 得到每个image文件对应的roi_mask文件的路径
    data_samples = dict()
    for i in os.listdir(stent_path): # 类似于 ICBM_010_sample_00.nii.gz
        sample_name = i.split('_sample_')[0]
        # sample_index = int(i.split('_sample_')[-1].split('_')[0])
        if(data_samples.get(sample_name) is None):
            data_samples[sample_name] = [i]
        else:
            data_samples[sample_name].append(i)
    
    for phantom in tqdm(sorted(os.listdir(image_path))): #类似于ICBM_010.nii.gz
        if(split_json is not None):
            if(phantom.split('.nii.gz')[0] not in split_json.keys()):
                print(f"{phantom} not in to be process, continue")
                continue
        ## 读取数据
        phantom_image_nib = nib.load(join(image_path, phantom))

        phantom_image_data = phantom_image_nib.get_fdata().astype(np.float32)
        phantom_image_data = hu_to_mu(phantom_image_data) #转换为hu
        # stent_insert_value = np.random.uniform(0.07, 0.15) #对于cta来说，这个是较为合适的，不是很明显，也不是不明显
        stent_insert_value = np.random.uniform(0.05, 0.10) #保持和原始的一致

        ## 每个数据有10个采样点
        # stent_insert_value = np.max(phantom_image_nib.get_fdata()) * 1.5
        sample_points = data_samples.get(phantom.split('.nii.gz')[0])
        if(sample_points is None):
            print(phantom, ' 在vol_gt_mask中不存在')
            continue
        for sample_point in sample_points: #0~9  # 类似于ICBM_010_sample_01_000.nii.gz
            phantom_image_data_with_stent = np.copy(phantom_image_data)

            ## 读取stent和roi mask
            stent_nib = nib.load(join(stent_path, sample_point))
            stent_data = np.round(stent_nib.get_fdata()).astype(np.int8)
            marker_data = None
            if marker_path is not None:
                marker_file = join(marker_path, sample_point)
                if not os.path.exists(marker_file):
                    raise FileNotFoundError(
                        f"marker_path was provided but {marker_file} does not exist"
                    )
                marker_nib = nib.load(marker_file)
                marker_data = np.round(marker_nib.get_fdata()).astype(np.int8)
                assert marker_nib.shape == stent_data.shape
            # roi_mask_data = np.round(nib.load(join(vol_gt_mask_path, sample_point)).get_fdata()).astype(np.int8)
            roi_proj = load_roi_mask_proj(join(vol_gt_mask_path, sample_point.split('.nii.gz')[0]))

            # assert phantom_image_nib.shape==stent_data.shape and roi_mask_data.shape==phantom_image_nib.shape
            assert phantom_image_nib.shape==stent_data.shape
            phantom_image_data_with_stent, label_data = compose_stent_and_marker(
                base_image=phantom_image_data,
                stent_mask=stent_data,
                marker_mask=marker_data,
                stent_value=stent_insert_value,
                marker_mu_ratio=marker_mu_ratio,
            )
            
            ## 还要对其进行投影
            x_dim_max, y_dim_max, z_dim_max = phantom_image_nib.shape[0], phantom_image_nib.shape[1], phantom_image_nib.shape[2]
            [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in  phantom_image_nib.header.get_zooms()[:3]]
            # u_pixel_size, v_pixel_size = x_voxel_size, y_voxel_size
            # u_detector, v_detector = 2048, 2048
            
            u_pixel_size, v_pixel_size = 0.308, 0.308
            u_detector, v_detector = 1500, 1500
            geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]), 
                    dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                    nDetector=np.array([u_detector, v_detector]),
                    dDetector=np.array([u_pixel_size, v_pixel_size]))
            angles = np.array(np.linspace(0, np.deg2rad(360), 361)[0:-1])

            ## 得到投影
            image_proj = tigre.Ax(np.transpose(phantom_image_data_with_stent, [2, 0, 1]), geo, angles)
            # roi_proj = tigre.Ax(np.transpose(roi_mask_data.astype(np.float32), [2, 0, 1]), geo, angles)
            label_proj = tigre.Ax(np.transpose(label_data, [2, 0, 1]), geo, angles)

            ## 写入文件
            nib.save(nib.Nifti1Image(phantom_image_data_with_stent, phantom_image_nib.affine, phantom_image_nib.header), join(stent_within_image_save_dir, sample_point))
            nib.save(
                nib.Nifti1Image(label_data, stent_nib.affine, stent_nib.header),
                join(stent_within_label_save_dir, sample_point),
            )
            tar_num = 360
            tar_index = list(np.linspace(0, 360, tar_num+1)[0:-1].astype(np.int16))

            for i in tar_index:
                roi_proj_slice = roi_proj[i]
                image_proj_slice = image_proj[i]
                label_proj_slice = label_proj[i]
                coords = np.where(roi_proj_slice > 0)
                x_min, x_max, y_min, y_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])

                rec_image_ = image_proj_slice[x_min:x_max+1, y_min:y_max +1] #截一个正方形的
                rec_label = label_proj_slice[x_min:x_max+1, y_min:y_max +1] #截一个正方形的
                
                image_rec_all_mask = np.zeros_like(image_proj_slice, dtype=np.float32)
                image_rec_all_mask[x_min:x_max+1, y_min:y_max +1] = 1
                rec_all_image = image_proj_slice * image_rec_all_mask
                rec_all_label = label_proj_slice * image_rec_all_mask

                cir_all_image = image_proj_slice * roi_proj_slice #得到roi投影
                cir_all_label = label_proj_slice * roi_proj_slice #得到roi投影

                cir_image = cir_all_image[x_min:x_max+1, y_min:y_max +1] #截一个圆形的
                cir_label = cir_all_label[x_min:x_max+1, y_min:y_max +1] #截一个圆形的

                write_sp(image=rec_image_, save_dir=join(rec_image_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=rec_all_image, save_dir=join(rec_all_image_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=cir_all_image, save_dir=join(cir_all_image_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=cir_image, save_dir=join(cir_image_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}_0000.npz'))
                
                write_sp(image=rec_label, save_dir=join(rec_label_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=rec_all_label, save_dir=join(rec_all_label_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=cir_all_label, save_dir=join(cir_all_label_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=cir_label, save_dir=join(cir_label_proj_save_dir, sample_point.split('.nii.gz')[0], sample_point.split('.nii.gz')[0]+f'_{str(int(i)).zfill(3)}.npz'))

### 这个文件的作用: 根据已经得到的全投影, 得到vol_gt_mask 投影所对应的image和label的部分

# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
#     parser.add_argument("--radius", type=int, default=64)
#     parser.add_argument("--resolution", type=float, default=0.2)
#     parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
#     parser.add_argument("--train", type=int, default=True) #0,1
#     parser.add_argument("--test", type=int, default=True)# 0,1
#     parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
#     parser.add_argument("--split", type=int, required=True)
#     args = parser.parse_args()

#     radius = args.radius
#     resolution = args.resolution
#     data_dir = args.data_dir
#     data_name = args.data_name
#     to_train = args.train
#     to_test = args.test
#     split = args.split
#     assert to_train in set([0,1])
#     assert to_test in set([0,1])
    
#     data_dir = join(data_dir, data_name)

#     print(f'data_dir: {data_dir}\n'
#           f'radius: {radius}\n',
#           f'resolution: {resolution}\n')


    # if(to_train):
    #     print('生成训练数据')
    #     mode='train'
    #     image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
    #     stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
    #     vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data/')
    #     save_dir = join(data_dir, f'voxel_{resolution}/proj/stent/radius-{radius}/{mode}_data')
    #     json_path = join('/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/split', mode, f"part_{split}.json")
    #     gen_roi_proj(image_path=image_path, stent_path=stent_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, json_path = json_path)

    # if(to_test):
    #     print('生成测试数据')
    #     mode='test'
    #     image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
    #     stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
    #     vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data/')
    #     save_dir = join(data_dir, f'voxel_{resolution}/proj/stent/radius-{radius}/{mode}_data')
    #     json_path = join('/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/split', mode, f"part_{split}.json")
    #     gen_roi_proj(image_path=image_path, stent_path=stent_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, json_path = json_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--marker_path", type=str, default=None)
    parser.add_argument("--marker_mu_ratio", type=float, default=2.0)
    parser.add_argument("--output_name", type=str, default="stent")
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    data_dir = args.data_dir
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    split = args.split
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    
    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n'
          f'radius: {radius}\n',
          f'resolution: {resolution}\n')


    if(to_train):
        print('生成训练数据')
        mode='train'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
        stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/{args.output_name}/radius-{radius}/{mode}_data')
        json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            split_json = json.load(f)
        gen_roi_proj(image_path=image_path, stent_path=stent_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, split_json=split_json, marker_path=args.marker_path, marker_mu_ratio=args.marker_mu_ratio)

    if(to_test):
        print('生成测试数据')
        mode='test'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
        stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/stent/radius-{radius}/{mode}_data')
        # json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
        gen_roi_proj(image_path=image_path, stent_path=stent_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, split_json=None, marker_path=args.marker_path, marker_mu_ratio=args.marker_mu_ratio)





