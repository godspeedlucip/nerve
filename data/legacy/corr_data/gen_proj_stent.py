import numpy as np
import plyfile, copy, pickle, os, random, json, argparse, shutil
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

from utils.file_rw import read_pkl, save_sparse_projections


from utils.general import set_geo
    

# 对于支架来说，缺的就是对支架的直接投影
def main(data_dir, save_dir = './',  radius=60, already_has = None, resolution = (0.2, 0.2, 0.2)):
    proj_gt_save_path = join(save_dir,'proj_gt')
    os.makedirs(proj_gt_save_path, exist_ok=True)

    stent_data_dir = join(data_dir, 'stent')

    for i in tqdm(os.listdir(stent_data_dir)):
        stent_nib = nib.load(join(stent_data_dir, i))

        angles = np.linspace(0, 2*np.pi, 360)
        radius = radius
        x_dim_max, y_dim_max, z_dim_max = stent_nib.shape[0], stent_nib.shape[1], stent_nib.shape[2], 
        x_voxel_size, y_voxel_size, z_voxel_size = [round(i, 4) for i in  stent_nib.header['pixdim'][1:4]] #voxel space=
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array([1024*2, 1024*2]), #投影要大一点，因为重采样后的数据会很大
            dDetector=np.array([x_voxel_size, y_voxel_size]))

        src_stent_proj = tigre.Ax(np.transpose(stent_nib.get_fdata().astype(np.float32), [2, 0, 1]), geo, angles) #原始图像的投影
        save_sparse_projections(proj=src_stent_proj, out_dir=join(proj_gt_save_path, i))


if __name__=="__main__":
    ## get_corr_stent生成的还只有图像，还需要生成和血管一样的src_data. 目前来说, gt_mask已经和血管一样, vol_gt就是这样的数据. 因此需要生成的就是proj了
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_dir",type=str, default='/data2/liuyang/liuyang/stent/interior/corr_data/data')
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--radius", type=int, default=64)
    args = parser.parse_args()

    data_dir = args.data_dir
    # stent_path = args.stent_path
    resulotion = args.resolution
    radius = args.radius

    ## 生成训练和测试数据
    print('生成训练数据')
    mode='train'
    final_data_dir = join(data_dir, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data')
    final_save_dir = final_data_dir
    main(data_dir=final_data_dir, save_dir=final_save_dir, radius=radius)

    ## 生成训练和测试数据
    print('生成测试数据')
    mode='test'
    final_data_dir = join(data_dir, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data')
    final_save_dir = final_data_dir
    main(data_dir=final_data_dir, save_dir=final_save_dir, radius=radius)

