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

def load_roi_mask_proj(proj_dir):
    proj = []
    for i in sorted(os.listdir(proj_dir)):
        proj.append(read_sp(join(proj_dir, i)))
    return np.stack(proj, axis=0)

def gen_roi_proj(image_path, label_path, vol_gt_mask_path, save_dir, split_json):
    ## 创建输出路径

    ct_vessel_image_save_path = join(save_dir, 'ct_vessel_image_2') #转换之后的图像
    cir_all_image_proj_save_dir = join(save_dir, 'image_proj', 'cir_all')
    rec_all_image_proj_save_dir = join(save_dir, 'image_proj', 'rec_all')
    rec_image_proj_save_dir = join(save_dir, 'image_proj', 'rec')
    cir_image_proj_save_dir = join(save_dir, 'image_proj', 'cir')
    cir_all_label_proj_save_dir = join(save_dir, 'label_proj', 'cir_all')
    rec_all_label_proj_save_dir = join(save_dir, 'label_proj', 'rec_all')
    rec_label_proj_save_dir = join(save_dir, 'label_proj', 'rec')
    cir_label_proj_save_dir = join(save_dir, 'label_proj', 'cir')
    os.makedirs(ct_vessel_image_save_path, exist_ok=True)
    
    ## 得到每个image文件对应的roi_mask文件的路径
    data_samples = dict()
    for i in os.listdir(vol_gt_mask_path): # 类似于 ICBM_010_sample_00
        sample_name = i.split('_sample_')[0] # 类似于ICBM_010
        if(data_samples.get(sample_name) is None):
            data_samples[sample_name] = [i]
        else:
            data_samples[sample_name].append(i)
    
    for phantom in tqdm(sorted(os.listdir(image_path))): #类似于ICBM_010.nii.gz
        if split_json is not None:
            if phantom.split('.nii.gz')[0] not in split_json.keys():
                print(f"{phantom} not in to be process, continue")
                continue
        # if(phantom.split('.nii.gz')[0] not in split_json.keys()):
        #     print(f"{phantom} not in to be process, continue")
        #     continue

        ## 读取数据
        phantom_image_nib = nib.load(join(image_path, phantom))
        # phantom_image_data = coarse_mri_to_ct(phantom_image_nib)
        phantom_image_data = phantom_image_nib.get_fdata().astype(np.float32)
        phantom_image_data = hu_to_mu(phantom_image_data)
        phantom_label_nib = nib.load(join(label_path, phantom))
        phantom_label_data = np.round(phantom_label_nib.get_fdata()).astype(np.int16)
        # ves_insert_value = 0.9 # 造影剂的衰减系数是这样的
        ves_insert_value = np.random.uniform(0.06, 0.11) # 对于cta数据来说，这个值比较好，因为血管本来就是会比较明显，但是这个又不是说一点都不明显
        
        ## 每个数据有10个采样点
        sample_points = data_samples.get(phantom.split('.nii.gz')[0]) 
        if(sample_points is None):
            print(phantom, ' 在vol_gt_mask中不存在')
            continue
        
        phantom_image_data[phantom_label_data > 0] = ves_insert_value
        nib.save(nib.Nifti1Image(phantom_image_data, phantom_image_nib.affine, phantom_image_nib.header), join(ct_vessel_image_save_path, phantom))

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
        image_proj = tigre.Ax(np.transpose(phantom_image_data.astype(np.float32), [2, 0, 1]), geo, angles)
        label_proj = tigre.Ax(np.transpose(phantom_label_data.astype(np.float32), [2, 0, 1]), geo, angles)

        for sample_point in sample_points: #0~9  # 类似于ICBM_010_sample_01_000
            roi_proj = load_roi_mask_proj(join(vol_gt_mask_path, sample_point.split('.nii.gz')[0]))
            tar_num = 360
            tar_index = list(np.round(np.linspace(0, 360, tar_num+1)[0:-1]).astype(np.int16))

            for i in tar_index:
                roi_proj_slice = roi_proj[i]
                image_proj_slice = image_proj[i]
                label_proj_slice = label_proj[i]
                coords = np.where(roi_proj_slice > 0)
                x_min, x_max, y_min, y_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
                image_rec = image_proj_slice[x_min:x_max+1, y_min:y_max +1] #截一个正方形的
                label_rec = label_proj_slice[x_min:x_max+1, y_min:y_max +1] #截一个正方形的

                image_rec_all_mask = np.zeros_like(image_proj_slice, dtype=np.float32)
                image_rec_all_mask[x_min:x_max+1, y_min:y_max +1] = 1
                image_rec_all = image_proj_slice * image_rec_all_mask
                label_rec_all = label_proj_slice * image_rec_all_mask

                image_cir_all = image_proj_slice * roi_proj_slice #得到roi投影
                label_cir_all = label_proj_slice * roi_proj_slice #得到roi投影

                image_cir = image_cir_all[x_min:x_max+1, y_min:y_max +1] #截一个圆形的
                label_cir = label_cir_all[x_min:x_max+1, y_min:y_max +1] #截一个圆形的

                write_sp(image=image_cir_all, save_dir=join(cir_all_image_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_rec_all, save_dir=join(rec_all_image_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_cir, save_dir=join(cir_image_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_rec, save_dir=join(rec_image_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}_0000.npz'))
                
                write_sp(image=label_cir_all, save_dir=join(cir_all_label_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_rec_all, save_dir=join(rec_all_label_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_cir, save_dir=join(cir_label_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_rec, save_dir=join(rec_label_proj_save_dir, sample_point, sample_point+f'_{str(int(i)).zfill(3)}.npz'))

# ### 这个文件的作用: 根据已经得到的全投影, 得到vol_gt_mask 投影所对应的image和label的部分
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

#     if(to_train):
#         print('生成训练数据')
#         mode='train'
#         image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
#         label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
#         vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
#         save_dir = join(data_dir, f'voxel_{resolution}/proj/vessel/radius-{radius}/{mode}_data')
#         json_path = join(f'/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/split/{mode}', f"part_{split}.json")
#         gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, json_path = json_path)

#     if(to_test):
#         print('生成测试数据')
#         mode='test'
#         image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
#         label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
#         vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
#         save_dir = join(data_dir, f'voxel_{resolution}/proj/vessel/radius-{radius}/{mode}_data')
#         json_path = join(f'/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/split/{mode}', f"part_{split}.json")
#         gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, json_path = json_path)




### 这个文件的作用: 根据已经得到的全投影, 得到vol_gt_mask 投影所对应的image和label的部分
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--use_split", type=int, default=True)
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    data_dir = args.data_dir
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    split = args.split
    use_split = args.use_split
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    
    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n'
          f'radius: {radius}\n',
          f'resolution: {resolution}\n')

    if(to_train):
        print('生成训练数据')
        mode='train'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/image')
        label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/vessel/radius-{radius}/{mode}_data')
        if(use_split):
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, split_json = split_json)

    if(to_test):
        print('生成测试数据')
        mode='test'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/image')
        label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/vessel/radius-{radius}/{mode}_data')
        if(use_split):
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path, save_dir=save_dir, split_json = split_json)

