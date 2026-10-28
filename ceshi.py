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
from utils.general import set_geo, voxelize, coarse_mri_to_ct, add_skull_box_simple

if __name__=='__main__':
    mri_path = "/data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/src_data/radius-64/vessel/train_data/fill_hole_image/ICBM_010.nii.gz"
    ct_save_path = "/data1/liuyang/stent/DTE/gen_data/other/ct_data.nii.gz"
    ## 读取数据
    phantom_image_nib = nib.load(join(mri_path))
    phantom_image_data = phantom_image_nib.get_fdata().astype(np.float32)
    phantom_image_data = coarse_mri_to_ct(phantom_image_data) # 转换为合理的ct
    phantom_image_data = add_skull_box_simple(phantom_image_data, skull_mu_min=0.01, skull_mu_max=0.015) #对于topbrain_mr来说,他本身是有骨头的. 因此只需要再加一个0.01~0.015就行了
    nib.save(nib.Nifti1Image(phantom_image_data, phantom_image_nib.affine, phantom_image_nib.header), ct_save_path)


