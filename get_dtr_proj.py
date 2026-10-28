import numpy as np
import plyfile, copy, pickle, os, random, json, argparse, shutil
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


def gen_dtr_proj(image_path, label_path, vol_gt_mask_path, save_path, tar_range=60, tar_num=3):
    # target_num = 18
    # tar_index = np.linspace(0, 360, target_num+1)[0:-1].astype(np.int16)

    ranges = int(180 / tar_range)
    ranges_to_chose = ranges
    to_train_index = []
    if(ranges > 10):
        ranges_to_chose = 10 #最多采集10个吧，太多了也不行
    for j in random.sample(range(ranges), k=ranges_to_chose):
        index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.int16))
        to_train_index.extend(index)
    to_train_index = set(sorted(list(set(to_train_index)))) 
    print(to_train_index)
    dest_label_path = join(save_path, 'label')
    dest_image_path = join(save_path, 'image')
    dest_vol_gt_path = join(save_path, 'vol_gt_proj')
    os.makedirs(dest_vol_gt_path, exist_ok=True)
    os.makedirs(dest_image_path, exist_ok=True)
    # 移动label
    src_label_save_path = join(dest_label_path, 'src_label')
    binary_label_save_path = join(dest_label_path, 'bin_label')
    os.makedirs(src_label_save_path, exist_ok=True)
    os.makedirs(binary_label_save_path, exist_ok=True)
    for i in tqdm(os.listdir(label_path)): #i=mr_001_sample_00

        # if(i not in to_process):
        #     continue

        file_list = sorted(os.listdir(join(label_path, i)))
        for j in to_train_index:
            data_path = file_list[j]
            src_label = read_sp(join(label_path, i, data_path))
            binary_label = np.copy(src_label)
            binary_label[src_label > 0.001] = 1
            binary_label[src_label <= 0.001] = 0
            binary_label = binary_label.astype(np.int8)
            j_name = data_path.replace('_','-')
            j_name = j_name.split('.npz')[0]+'.tif'
            tiff.imwrite(join(src_label_save_path, j_name), src_label.astype(np.float32))
            tiff.imwrite(join(binary_label_save_path, j_name), binary_label)

    # 移动image
    for i in tqdm(os.listdir(image_path)): #mr_001_sample_00

        # if(i not in to_process):
        #     continue

        file_list = sorted(os.listdir(join(image_path, i)))
        for j in to_train_index:
            data_path = file_list[j]
            image = read_sp(join(image_path, i, data_path))
            j_name = data_path.replace('_','-')
            j_name = j_name.split('-0000')[0]+'_0000.tif'
            tiff.imwrite(join(dest_image_path, j_name), image.astype(np.float32))

    ## 移动vol_gt_mask_proj
    for i in tqdm(os.listdir(vol_gt_mask_path)): #mr_001_sample_00

        # if(i not in to_process):
        #     continue

        file_list = sorted(os.listdir(join(vol_gt_mask_proj_path, i)))
        for j in to_train_index:
            data_path = file_list[j]
            mask_proj = read_sp(join(vol_gt_mask_proj_path, i, data_path))
            mask_proj[mask_proj>0.001] = 1
            mask_proj[mask_proj<=0.001] = 0
            mask_proj = mask_proj.astype(np.uint8)
            j_name = data_path.replace('_','-')
            j_name = j_name.split('.npz')[0]+'.tif'
            tiff.imwrite(join(dest_vol_gt_path, j_name), mask_proj)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range", type=int, default=60)
    parser.add_argument("--tar_num", type=int, default=3)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_name", type=str, default='costa_new') #costa、topbrain
    parser.add_argument("--data_type", type=str, default='vessel') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    tar_range = args.tar_range
    tar_num = args.tar_num
    data_dir = args.data_dir
    data_name = args.data_name
    data_type = args.data_type
    to_train = args.train
    to_test = args.test
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    
    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n'
          f'radius: {radius}\n',
          f'resolution: {resolution}\n')

    ## 测试时用的dtr
    if(to_train):
        print('生成训练数据')
        mode='train'
        image_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/image_proj/rec')
        label_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/label_proj/rec')

        # vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj_2/vol_gt_mask_proj/radius-{radius}/{mode}_data')

        # save_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/dtr/new_proj_2/{data_type}/radius-{radius}/{mode}_data')
        gen_dtr_proj(image_path=image_path, label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, tar_range=tar_range, tar_num=tar_num)

    if(to_test):
        print('生成测试数据')
        mode='test'
        image_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/image_proj/rec')
        label_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/label_proj/rec')
        # vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj_2/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        # save_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/dtr/new_proj_2/{data_type}/radius-{radius}/{mode}_data')
        gen_dtr_proj(image_path=image_path, label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, tar_range=tar_range, tar_num=tar_num)


    ### 测试用的full_dtr的存储位置
    # if(to_train):
    #     print('生成训练数据')
    #     mode='train'
    #     image_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/image_proj/rec')
    #     label_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/label_proj/rec')
    #     vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
    #     save_path = join(data_dir, f'voxel_{resolution}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data')
    #     gen_dtr_proj(image_path=image_path, label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, tar_range=tar_range, tar_num=tar_num)

    # if(to_test):
    #     print('生成测试数据')
    #     mode='test'
    #     image_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/image_proj/rec')
    #     label_path = join(data_dir, f'voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data/label_proj/rec')
    #     vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
    #     save_path = join(data_dir, f'voxel_{resolution}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data')
    #     gen_dtr_proj(image_path=image_path, label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, tar_range=tar_range, tar_num=tar_num)



