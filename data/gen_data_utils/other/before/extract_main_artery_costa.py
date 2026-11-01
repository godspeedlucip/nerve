from scipy.ndimage import distance_transform_edt, binary_dilation, label, find_objects
import numpy as np
import nibabel as nib
import os
join = os.path.join


def extract_and_denoise_large_vessels_from_nii(
    input_nii_img: nib.Nifti1Image, 
    radius_threshold_voxels: float,
    min_volume_voxels: int,  # **新增参数：最小连通区域体积阈值 (体素数量)**
    dilation_iterations: int = 1
) -> nib.Nifti1Image:
    """
    从 NIfTI 格式的血管掩膜对象中提取直径较大的血管部分，恢复边缘，并去除小体积的离散噪声。

    参数:
    input_nii_img (nib.Nifti1Image): 输入的三维二值血管掩膜的 NIfTI 图像对象。
    radius_threshold_voxels (float): 用于提取大血管中心骨架的半径阈值（体素单位）。
    min_volume_voxels (int): 最小连通区域的体素数量阈值。小于此体积的区域将被视为噪声去除。
    dilation_iterations (int): 膨胀操作的迭代次数。默认值 1。

    返回:
    nib.Nifti1Image: 一个新的 NIfTI 图像对象，只包含去噪后的大血管主体。
    """
    
    # 1. 提取数据和元数据
    vessel_mask_3d = input_nii_img.get_fdata()
    affine = input_nii_img.affine
    header = input_nii_img.header
    
    # 确保数据是二值掩膜 (True/False)
    binary_mask = vessel_mask_3d > 0

    # 2. 欧氏距离变换 (EDT) 和 阈值化 - 提取大血管中心骨架
    distance_map = distance_transform_edt(binary_mask)
    large_vessel_skeleton = distance_map >= radius_threshold_voxels
    
    # 3. 膨胀操作 (Dilation) - 恢复血管边缘
    struct = np.ones((3, 3, 3), dtype=bool) 
    dilated_skeleton = binary_dilation(
        large_vessel_skeleton,
        structure=struct,
        iterations=dilation_iterations
    )
    
    # 4. 限制膨胀后的结果在原始血管掩膜内，得到初步的大血管掩膜
    large_artery_mask_preliminary = dilated_skeleton & binary_mask
    
    # 5. **【新增步骤】** 连通域分析去噪
    
    # a. 标记连通区域
    # labeled_array: 标记后的数组 (背景为 0，第一个区域为 1，第二个为 2, ...)
    # num_features: 连通区域的总数
    labeled_array, num_features = label(large_artery_mask_preliminary)
    
    # print(f"在去噪前，共找到 {num_features} 个连通区域。")
    
    # b. 计算每个区域的体积并构建筛选列表
    # 列表 index 0 对应 label 1 的体积，以此类推
    component_sizes = np.bincount(labeled_array.flat)[1:]
    
    # 找出所有体积大于阈值的区域的标签（label）
    # np.where 返回的数组是 0-indexed，所以需要加 1 得到实际的 label 值
    valid_labels = np.where(component_sizes >= min_volume_voxels)[0] + 1
    
    # c. 创建去噪后的最终掩膜
    # 初始化一个零数组作为最终结果
    large_artery_mask_denoised = np.zeros_like(vessel_mask_3d, dtype=np.uint8)
    
    # 遍历有效标签，并将对应区域复制到最终掩膜
    for lbl in valid_labels:
        # 使用逻辑或 (|) 将符合条件的区域添加到最终掩膜中
        large_artery_mask_denoised[labeled_array == lbl] = 1
    
    # print(f"成功保留 {len(valid_labels)} 个连通区域。")

    # 6. 创建并返回新的 NIfTI 图像对象
    new_nii_img = nib.Nifti1Image(large_artery_mask_denoised, affine, header=header)
    new_nii_img.set_data_dtype(np.uint8)
    
    return new_nii_img


if __name__=='__main__':
    ## 尝试对所有文件都进行主动脉提取
    costa_data_dir = '/data1/liuyang/stent/interior/data/costa/root_data/test_data'
    save_dir = '/data1/liuyang/stent/interior/data/costa/main_artery_mask_dir/test_data/2'
    os.makedirs(save_dir,exist_ok=True)
    for i in os.listdir(costa_data_dir):
        mask_data = nib.load(join(costa_data_dir, i))
        main_artery_mask = extract_and_denoise_large_vessels_from_nii(mask_data, radius_threshold_voxels=2, min_volume_voxels=400, dilation_iterations=2)
        nib.save(main_artery_mask, join(save_dir,i))
        print(i)