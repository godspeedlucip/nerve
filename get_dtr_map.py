## 原始的trun_data的代码，和gen_truncate_data相比，我觉得与该是一样的，但是具体的也没看


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


def get_dte_map_input(pred_label_path, src_label_path, save_label_path):
    # pred_files = sorted([i for i in os.listdir(pred_label_path) if i.endswith('.tif')])
    # src_files = sorted([i for i in os.listdir(src_label_path) if i.endswith('.tif')])
    # for i in tqdm(range(len(pred_files))):
    #     # print(i)
    #     pred = tiff.imread(join(pred_label_path, pred_files[i])).astype(np.uint8)
    #     src = tiff.imread(join(src_label_path, src_files[i]))
    #     dte_map_input = src * pred
    #     tiff.imwrite(join(save_label_path, pred_files[i]), dte_map_input)
        for i in tqdm(os.listdir(pred_label_path)):
            if not i.endswith('.tif'):
                 continue
            pred = tiff.imread(join(pred_label_path, i)).astype(np.uint8)
            src = tiff.imread(join(src_label_path, i.split('.tif')[0]+'_0000.tif'))
            dte_map_input = src * pred
            tiff.imwrite(join(save_label_path, i), dte_map_input)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_path", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--data_name", type=str, default='costa_new')
    parser.add_argument("--data_type", type=str, default='vessel')
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()

    radius = args.radius
    resulotion = args.resolution
    data_path = args.data_path
    data_name = args.data_name
    data_type = args.data_type
    train = args.train
    test = args.test
    

    # 生成训练数据
    if(train):
        print('gen train data: ')
        mode='train'
        pred_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/bin_label')
        save_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/src_label')
        src_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/image')
        os.makedirs(save_label_path, exist_ok=True)
        get_dte_map_input(pred_label_path, src_label_path, save_label_path)

    # 之后生成测试数据
    if(test):
        print('gen test data: ')
        mode='test'
        pred_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/bin_label')
        save_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/src_label')
        src_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/dtr/proj/{data_type}/radius-{radius}/{mode}_data/image')
        os.makedirs(save_label_path, exist_ok=True)
        get_dte_map_input(pred_label_path, src_label_path, save_label_path)




    # # 之后生成测试数据
    # if(train):
    #     print('gen train data: ')
    #     mode='train'
    #     pred_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/bin_label')
    #     save_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/src_label')
    #     src_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/image')
    #     os.makedirs(save_label_path, exist_ok=True)
    #     get_dte_map_input(pred_label_path, src_label_path, save_label_path)

    # # 之后生成测试数据
    # if(test):
    #     print('gen test data: ')
    #     mode='test'
    #     pred_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/bin_label')
    #     save_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/predict/src_label')
    #     src_label_path = join(data_path ,f'{data_name}/voxel_{resulotion}/full_dtr/proj/{data_type}/radius-{radius}/{mode}_data/image')
    #     os.makedirs(save_label_path, exist_ok=True)
    #     get_dte_map_input(pred_label_path, src_label_path, save_label_path)


