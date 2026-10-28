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

from utils.file_rw import write_sp, read_sp
## 这段代码暂时不明白到底是干什么的. 好像没有用?

def gen_vol_gt(src_label_path, vol_gt_mask_path, save_path, split_json):
    vol_gt_save_path = join(save_path, 'vol_gt')
    vol_gt_no_mask_save_path = join(save_path, 'vol_gt_no_mask')
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(vol_gt_no_mask_save_path, exist_ok=True)

    info_dict = dict()
    for i in os.listdir(vol_gt_mask_path): #i: ICBM_010_sample_00.nii.gz
        if(not i.endswith('.nii.gz')):
            continue
        image_name = i.split('_sample_')[0] #ICBM_010_sample_00
        if(info_dict.get(image_name) is None):
            info_dict[image_name] = [i]
        else:
            info_dict[image_name].append(i)

    for i in tqdm(os.listdir(src_label_path)): #i: ICBM_010.nii.gz
        labels = info_dict.get(i.split('.nii.gz')[0]) #ICBM_010
        if(labels is None):
            continue

        if(split_json is not None):
            if(i.split('.nii.gz')[0] not in split_json.keys()):
                print(f"{i} not in to be process, continue")
                continue
        
        nib_src_label = nib.load(join(src_label_path, i))
        src_label = nib_src_label.get_fdata().astype(np.float32)
        for label in labels:
            vol_gt_mask = nib.load(join(vol_gt_mask_path, label)).get_fdata().astype(np.float32)
            vol_temp = np.zeros_like(vol_gt_mask, dtype=np.uint8)
            coords = np.where(vol_gt_mask>0)
            x_min, x_max, y_min, y_max, z_min, z_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1]), np.min(coords[2]), np.max(coords[2])
            vol_temp[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
            vol_label_no_mask = src_label * vol_temp
            vol_label = src_label * vol_gt_mask
            nib.save(nib.Nifti1Image(vol_label_no_mask, nib_src_label.affine, nib_src_label.header), join(vol_gt_no_mask_save_path, label))
            nib.save(nib.Nifti1Image(vol_label, nib_src_label.affine, nib_src_label.header), join(vol_gt_save_path, label))



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range", type=int, default=60)
    parser.add_argument("--tar_num", type=int, default=3)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_name", type=str, default='costa_new') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--split", type=int)
    parser.add_argument("--use_split", type=int)
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
        src_label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data')
        if(use_split):
            split_file = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
            with open(split_file, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_vol_gt(src_label_path=src_label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, split_json=split_json)

    if(to_test):
        print('生成测试数据')
        mode='test'
        src_label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data')
        if(use_split):
            split_file = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
            with open(split_file, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_vol_gt(src_label_path=src_label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path, split_json=split_json)




