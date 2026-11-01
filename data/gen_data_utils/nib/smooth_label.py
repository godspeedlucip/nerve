import SimpleITK as sitk
import os
from tqdm import tqdm
join=os.path.join

def smooth_label_via_sdf(
    label_img: sitk.Image,
    sdf_sigma_mm=0.3,     # 0.2–0.5 mm，越小越安全
    threshold_mm=0.0,
    closing_radius_vox=1 # 可选，0=不做
):
    """
    label_img: 二值label (0/1)
    """
    label = sitk.Cast(label_img > 0, sitk.sitkUInt8)

    # Signed distance field
    sdf = sitk.SignedMaurerDistanceMap(
        label,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True
    )
    sdf = sitk.Cast(sdf, sitk.sitkFloat32)

    # SDF 平滑（单位是 mm）
    if sdf_sigma_mm > 0:
        sdf = sitk.SmoothingRecursiveGaussian(sdf, sdf_sigma_mm)

    # 阈值回 label
    label_smooth = sitk.Cast(sdf > threshold_mm, sitk.sitkUInt8)

    # 可选：closing 修补极小断点
    if closing_radius_vox > 0:
        label_smooth = sitk.BinaryMorphologicalClosing(
            label_smooth, (closing_radius_vox,)*3
        )

    return label_smooth


if __name__=="__main__":
    ## 创建符合nnUNet格式的数据集
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--input_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args = parser.parse_args()
    
    save_dir = args.save_dir
    input_label = args.input_dir
    
    for i in tqdm(os.listdir(input_label)):
        smoo_label = smooth_label_via_sdf(sitk.ReadImage(join(input_label, i)))
        sitk.WriteImage(smoo_label, join(save_dir, i))



