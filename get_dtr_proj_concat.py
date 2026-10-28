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


def gen_dtr_proj(label_path, vol_gt_mask_path, save_path):
    info_dict = dict()
    for i in os.listdir(label_path): #i: ICBM-010-sample-00-000_pred.tif
        if(not i.endswith('.tif')):
            continue
        sample_name = i.split('_pred')[0][:-4] #ICBM-010-sample-00
        if(info_dict.get(sample_name) is None):
            info_dict[sample_name] = [i]
        else:
            info_dict[sample_name].append(i)

    for i in tqdm(os.listdir(vol_gt_mask_path)): #i: mr_001_sample_00
        # 其中的文件: ICBM_010_sample_00_000.npz
        i_re = i.replace('_','-')
        sampel_save_path = join(save_path, i)
        os.makedirs(sampel_save_path, exist_ok=True)

        # mask_projs = sorted(os.listdir(join(vol_gt_mask_path, i)))
        labels = info_dict.get(i_re)
        # print(labels)
        if(labels is None):
            continue
        for label in labels:
            rec_label = tiff.imread(join(label_path, label)).astype(np.float32)
            vol_gt_mask_proj = read_sp(join(vol_gt_mask_path, i, label.split('_pred')[0].replace('-','_')+'.npz'))
            vol_gt_mask_proj = vol_gt_mask_proj.astype(np.float32)

            coords = np.where(vol_gt_mask_proj > 0)
            x_min, x_max, y_min, y_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
            temp_mask = np.zeros_like(vol_gt_mask_proj, dtype=np.float32)
            temp_mask[x_min:x_max+1, y_min:y_max+1] = rec_label

            vol_gt_mask_proj = vol_gt_mask_proj * temp_mask #将值限制在圆形roi内

            write_sp(vol_gt_mask_proj, join(sampel_save_path, label.split('_pred')[0]+'.npz'))




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
    # tar_range = args.tar_range
    # tar_num = args.tar_num
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


    if(to_train):
        print('生成训练数据')
        mode='train'
        label_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/depth_label/result')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
        gen_dtr_proj(label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path)

    if(to_test):
        print('生成测试数据')
        mode='test'
        label_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/depth_label/result')
        vol_gt_mask_proj_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_path = join(data_dir, f'voxel_{resolution}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/complete_label')
        gen_dtr_proj(label_path=label_path, save_path=save_path, vol_gt_mask_path = vol_gt_mask_proj_path)




