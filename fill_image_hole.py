import numpy as np
import plyfile, copy, pickle, os, random, json, re, pickle, argparse
import pandas as pd
import tifffile as tiff
from scipy import io
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk
import tigre
import tigre.algorithms as algs
import nibabel as nib
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import zoom
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.ndimage import binary_dilation, generate_binary_structure



def expand_mask(mask, n_iters=3):
  struct = generate_binary_structure(3, 2)
  n_iters = 3
  mask_expanded = binary_dilation(mask, structure=struct, iterations=n_iters)
  return mask_expanded

def get_26_neighbors_offsets():
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets
NEIGHBOR_OFFSETS = get_26_neighbors_offsets()

def neighbor_weight(dx, dy, dz):
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist == 1:
        return 1.0          # face neighbor
    elif np.isclose(dist, np.sqrt(2)):
        return 0.7          # edge neighbor
    else:
        return 0.5          # corner neighbor

def has_nonmask_neighbor(mask, x, y, z):
    sx, sy, sz = mask.shape
    for dx, dy, dz in NEIGHBOR_OFFSETS:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
            if not mask[nx, ny, nz]:
                return True
    return False

def fill_mask_iterative(image, mask, max_iters=50):
    """
    image: float32/float64, shape (X, Y, Z)
    mask:  bool, True = need fill
    """
    image = image.copy()
    mask = mask.copy()

    sx, sy, sz = image.shape
    iteration = 0

    while mask.any():
        iteration += 1
        if iteration > max_iters:
            print("Warning: reached max iterations")
            break

        boundary_voxels = []

        # 找到当前 mask 的“边界层”
        coords = np.argwhere(mask)
        for x, y, z in coords:
            if has_nonmask_neighbor(mask, x, y, z):
                boundary_voxels.append((x, y, z))

        if len(boundary_voxels) == 0:
            print("No boundary voxels found, stopping.")
            break

        # 对边界体素进行填充
        for x, y, z in boundary_voxels:
            weighted_sum = 0.0
            weight_sum = 0.0

            for dx, dy, dz in NEIGHBOR_OFFSETS:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                    if not mask[nx, ny, nz]:
                        w = neighbor_weight(dx, dy, dz)
                        weighted_sum += w * image[nx, ny, nz]
                        weight_sum += w

            if weight_sum > 0:
                image[x, y, z] = weighted_sum / weight_sum
            else:
                image[x, y, z] = 0.0  # 理论上不应发生

        # 将已填充体素移出 mask
        for x, y, z in boundary_voxels:
            mask[x, y, z] = False

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, remaining voxels: {mask.sum()}")

    return image


def work(image_dir, label_dir, save_dir, mask_expand_n_iters=1):
    # expand_mask_save_dir = join(save_dir, f'mask_expand_{mask_expand_n_iters}')
    image_filled_save_dir = save_dir
    # os.makedirs(expand_mask_save_dir, exist_ok=True)
    os.makedirs(image_filled_save_dir, exist_ok=True)
    for i in tqdm(os.listdir(image_dir)):
        ## 读取数据
        image_nii = nib.load(join(image_dir, i))
        mask_nii = nib.load(join(label_dir, i))
        image_data = image_nii.get_fdata().astype(np.float32)
        mask_data = np.round(mask_nii.get_fdata()).astype(np.int16)
        mask_data[mask_data>0] = 1

        ## 获得扩张之后的mask
        mask_expand = expand_mask(mask_data, mask_expand_n_iters)
        mask_expand = np.round(mask_expand).astype(np.int16)

        # mask_expand = np.copy(mask_data)

        ## 填充image
        image_filled = fill_mask_iterative(image_data, mask_expand)
        
        ## 保存
        image_filled_nib = nib.Nifti1Image(image_filled, image_nii.affine, image_nii.header)
        # label_expand_nib = nib.Nifti1Image(mask_expand, mask_nii.affine, mask_nii.header)
        # nib.save(label_expand_nib, join(expand_mask_save_dir, i))
        nib.save(image_filled_nib, join(image_filled_save_dir, i))


### 以下代码是在src data中进行fill_hole的流程
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/src_data')
    args = parser.parse_args()

    data_dir = args.data_dir
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    assert to_train in set([0,1])
    assert to_test in set([0,1])

    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n',
        f'data_name: {data_name}\n')
    
    if(to_train):
        print('生成训练数据')
        mode='train'
        image_path = join(data_dir, f'{mode}_data/image')
        label_path = join(data_dir, f'{mode}_data/label')
        save_dir = join(data_dir, f'{mode}_data/fill_hole_image')
        work(image_dir = image_path, label_dir = label_path, save_dir = save_dir, mask_expand_n_iters=0)

    if(to_test):
        print('生成测试数据')
        mode='test'
        image_path = join(data_dir, f'{mode}_data/image')
        label_path = join(data_dir, f'{mode}_data/label')
        save_dir = join(data_dir, f'{mode}_data/fill_hole_image')
        work(image_dir = image_path, label_dir = label_path, save_dir = save_dir, mask_expand_n_iters=0)




