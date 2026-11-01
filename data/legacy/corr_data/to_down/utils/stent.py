
import numpy as np
import plyfile, copy, pickle, os, random, json, argparse, warnings
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
from gen_stent.utils.file_rw import read_ply
from gen_stent.utils.gen_stent import w_like_Stent, ring_Stent, Stent_Ply, sin_Stent

from utils.centerline import (
    extract_random_segment,
)

from utils.postprocess import (
    thicken_by_distance,
)


def get_segments(components, spacing, target_length, number_segments):
    indices = list(range(len(components)))
    random.shuffle(indices)
    valid_selected, segment_25mms = [],[]
    for idx in indices:
        segment_25mm = extract_random_segment(
            points_voxel=components[idx],
            spacing=spacing,
            length_mm=target_length #这里的length是物理长度
        )
        
        if segment_25mm is not None:
            segment_25mms.append(segment_25mm)
            valid_selected.append(components[idx])

        if(len(segment_25mms)==number_segments):
            break
    
    if(len(segment_25mms)==0):
        ## 如果完全不满足，就返回None
        return segment_25mms

    ## 就算选到重复的也要保证数量
    while(len(segment_25mms) != number_segments):
        indices = list(range(len(valid_selected)))
        random.shuffle(indices)
        idx = indices[0]
        segment_25mm = extract_random_segment(
            points_voxel=valid_selected[idx],
            spacing=spacing,
            length_mm=target_length #这里的length是物理长度
        )
        if segment_25mm is not None:
            segment_25mms.append(segment_25mm)
            valid_selected.append(valid_selected[idx])
    
    return segment_25mms

def get_stent_generator_train(ply_dir):
    stent1_generator = Stent_Ply(read_ply(join(ply_dir, 'stent1.ply')))
    stent2_generator = Stent_Ply(read_ply(join(ply_dir, 'stent2.ply')))
    stent3_generator = Stent_Ply(read_ply(join(ply_dir, 'stent3.ply')))
    stent4_generator = Stent_Ply(read_ply(join(ply_dir, 'stent4.ply')))
    wall_stent_generator = Stent_Ply(read_ply(join(ply_dir, 'wall_stent.ply')))
    sin_stent_generator = sin_Stent()
    w_like_stent_generator = w_like_Stent()
    ring_stent_generator = ring_Stent()

    stent_name = ['stent1', 'stent2', 'stent3', 'stent4', 'wallStent', 'sinStent', 'wLikeStent', 'ringStent']

    return [stent1_generator, stent2_generator, stent3_generator, stent4_generator, wall_stent_generator, sin_stent_generator, w_like_stent_generator, ring_stent_generator], stent_name

def get_stent_generator_test(ply_dir):
    stent5_generator = Stent_Ply(read_ply(join(ply_dir, 'stent5.ply')))
    w_stent_generator = Stent_Ply(read_ply(join(ply_dir, 'w_stent.ply')))
    stent_name = ['stent5', 'w_stent']
    return [stent5_generator, w_stent_generator], stent_name


def stent_expand(sub_voxel, stent_name):
    if(stent_name in set(['wall_stent'])): #这个支架太密集了，直接返回原始的数据就行了
        return sub_voxel, True
    else:
        expand_sub_voxel, sub_voxel_skeleton, flag = thicken_by_distance(sub_voxel, r_voxel = 1) #尝试扩大其半径
        return expand_sub_voxel, flag