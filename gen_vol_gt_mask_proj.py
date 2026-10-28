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
import scipy.sparse as sp

from utils.file_rw import load_3d_from_z_indices
from utils.general import set_geo, voxelize


def get_proj_dir(image_path, save_dir, type_of='image', split_json = None):
#   image_proj_save_path = join(save_dir, f'{type_of}_proj')
    image_proj_save_path = save_dir

    for i in tqdm(sorted(os.listdir(image_path))):
        if(split_json is not None):
            if(i.split('_sample_')[0] not in split_json.keys()):
                print('not in process: ', i)
                continue
        image_nib = nib.load(join(image_path, i))
        if(type_of=='image'):
            image_data = image_nib.get_fdata().astype(np.float32)
        else:
            image_data = np.round(image_nib.get_fdata().astype(np.float32)).astype(np.int16)

        x_dim_max, y_dim_max, z_dim_max = image_nib.shape[0], image_nib.shape[1], image_nib.shape[2]
        [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in  image_nib.header['pixdim'][1:4]]
        
        # u_pixel_size, v_pixel_size = x_voxel_size, y_voxel_size
        # u_detector, v_detector = 2048, 2048

        u_pixel_size, v_pixel_size = 0.308, 0.308
        u_detector, v_detector = 1500, 1500
        
        ## 设置投影几何
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]), 
                dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                nDetector=np.array([u_detector, v_detector]),
                dDetector=np.array([u_pixel_size, v_pixel_size]))
        angles = np.array(np.linspace(0, 2*np.pi, 361)[0:-1]) #取0, 1, 2, ..., 359

        image_proj = tigre.Ax(np.transpose(image_data.astype(np.float32), [2, 0, 1]), geo, angles)

        ## 尝试一下保存原始的投影部分
        if(type_of=='roi_mask'):
            image_proj[image_proj >= 0.01] = 1
            image_proj[image_proj < 0.01] = 0
            image_proj = image_proj.astype(np.int8)

        sample_name = i.split('.nii.gz')[0]
        ## 直接保存原始的投影，
        save_num = 360
        image_sample_proj_save_path = join(image_proj_save_path, sample_name)
        os.makedirs(image_sample_proj_save_path, exist_ok=True)
        for to_be_process, process_save_path in zip([image_proj], [image_sample_proj_save_path]):
            save_index = np.linspace(0, to_be_process.shape[0], save_num + 1)[0:-1].astype(np.int16)
            if(type_of=='image'):
                for k in save_index:
                    # 图像直接存为tif
                    save_file_name = sample_name + '_'+str(int(k)).zfill(3)+'_0000.tif' #2D数据直接存储为tif就行了。 类似于ICBM_010_sample_00_001.nii.gz
                    tiff.imwrite(join(process_save_path, save_file_name), to_be_process[k])
                    # np.savez_compressed(join(process_save_path, save_file_name), arr=to_be_process[k])

            else:
                for k in save_index:
                    # 用稀疏矩阵的压缩存储
                    save_file_name = sample_name + '_'+str(int(k)).zfill(3)+'.npz'
                    sp.save_npz(
                        join(process_save_path, save_file_name),
                        sp.csr_matrix(to_be_process[k]),
                        compressed=True
                    )

## 输入不同的文件路径和type_of, 得到其对应的全投影

# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
#     parser.add_argument("--radius", type=int, default=64)
#     parser.add_argument("--resolution", type=float, default='0.2')
#     parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
#     # parser.add_argument("--data_type", type=str, default='vessel') #stent vessel
#     parser.add_argument("--train", type=int, default=True) #0,1
#     parser.add_argument("--test", type=int, default=True)# 0,1
#     parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
#     parser.add_argument("--type_of", type=str, default='roi_mask')
#     parser.add_argument("--split_num", type=int, default=1)
#     args = parser.parse_args()

#     radius = args.radius
#     resolution = args.resolution
#     data_dir = args.data_dir
#     data_name = args.data_name
#     to_train = args.train
#     to_test = args.test
#     type_of = args.type_of
#     split_num = args.split_num
#     # assert data_type in set(['stent', 'vessel'])
#     assert to_train in set([0,1])
#     assert to_test in set([0,1])
#     assert type_of in set(['image', 'label', 'roi_mask'])
    
#     data_dir = join(data_dir, data_name)

#     print(f'data_dir: {data_dir}\n',
#         #   f'data_type: {data_type}\n',
#           f'radius: {radius}\n',
#           f'resolution: {resolution}\n',
#           f'type: {type_of}')

#     if(to_train):
#         print('生成训练数据')
#         mode='train'
#         image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt_mask')
#         save_dir = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data/')
#         already_has = None
#         get_proj_dir(image_path=image_path, save_dir=save_dir, type_of = type_of)

#     if(to_test):
#         print('生成测试数据')
#         mode='test'
#         image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/{data_type}/{mode}_data/vol_gt_mask')
#         save_dir = join(data_dir, f'voxel_{resolution}/proj/vol_gt_mask_proj/radius-{radius}/{mode}_data/')
#         already_has = None
#         get_proj_dir(image_path=image_path, save_dir=save_dir, type_of = type_of)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default='0.2')
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    # parser.add_argument("--data_type", type=str, default='vessel') #stent vessel
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--type_of", type=str, default='roi_mask')
    parser.add_argument("--split_num", type=int, default=1)
    parser.add_argument("--use_split", type=int, default=1)
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    data_dir = args.data_dir
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    type_of = args.type_of
    split_num = args.split_num
    use_split = args.use_split
    # assert data_type in set(['stent', 'vessel'])
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    assert type_of in set(['image', 'label', 'roi_mask'])
    
    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n',
        #   f'data_type: {data_type}\n',
          f'radius: {radius}\n',
          f'resolution: {resolution}\n',
          f'type: {type_of}')

    if(to_train):
        print('生成训练数据')
        mode='train'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        if use_split:
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split', mode, f"part_{split_num}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        get_proj_dir(image_path=image_path, save_dir=save_dir, type_of = type_of, split_json = split_json)

    if(to_test):
        print('生成测试数据')
        mode='test'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vol_gt_mask/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        if use_split:
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split', mode, f"part_{split_num}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        get_proj_dir(image_path=image_path, save_dir=save_dir, type_of = type_of, split_json = split_json)



