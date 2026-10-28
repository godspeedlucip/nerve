"""
基于 get_dtr_map / get_dtr_proj_concat / get_dtr_proj / get_dtr_recon 改编，
适配 Dataset001_STENT 的文件布局 (imagesTs / labelsTs / pred)。

功能:
  1. process_pred:  pred * image  →  DTE map input (来自预测结果)
  2. process_label: label * image →  DTE map input (来自真实标签)

原始逻辑来自 get_dtr_map.py 的 get_dte_map_input:
  - 对每个 pred/label tif，找到对应的 image tif
  - 逐元素相乘 src * pred (或 src * label)
  - 保存结果
"""

import numpy as np
import os
import argparse
import tifffile as tiff
from tqdm import tqdm

join = os.path.join


def process_map(pred_or_label_path, image_path, save_path):
    """
    对 pred (或 label) 目录中的每个 .tif，与 imagesTs 中对应的图像相乘，
    得到 DTE map input。

    Parameters
    ----------
    pred_or_label_path : str
        pred/ 或 labelsTs/ 的路径，文件命名: stent_001_XXXX.tif
    image_path : str
        imagesTs/ 的路径，文件命名: stent_001_XXXX_0000.tif
    save_path : str
        输出目录
    """
    os.makedirs(save_path, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(pred_or_label_path))):
        if not fname.endswith('.tif'):
            continue

        # pred/label 文件: stent_001_0031.tif
        # 对应 image 文件: stent_001_0031_0000.tif
        base = fname.split('.tif')[0]          # stent_001_0031
        image_fname = base + '_0000.tif'       # stent_001_0031_0000.tif
        image_file = join(image_path, image_fname)

        if not os.path.exists(image_file):
            print(f'  [跳过] 找不到对应图像: {image_file}')
            continue

        pred_or_label = tiff.imread(join(pred_or_label_path, fname)).astype(np.uint8)
        src = tiff.imread(image_file)

        dte_map_input = src * pred_or_label

        tiff.imwrite(join(save_path, fname), dte_map_input)


def main():
    parser = argparse.ArgumentParser(
        description="对 Dataset001_STENT 的 pred 和 label 做后处理，生成 DTE map input"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default='/data1/liuyang/stent/interior/real_data/Dataset001_STENT',
        help="Dataset001_STENT 根目录"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default='/data1/liuyang/stent/interior/real_data/Dataset001_STENT/dtr_post',
        help="输出根目录"
    )
    parser.add_argument("--process_pred", type=int, default=1, help="是否处理 pred (1/0)")
    parser.add_argument("--process_label", type=int, default=1, help="是否处理 labelTs (1/0)")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    image_path = join(data_dir, 'imagesTs')

    # ---------- 处理 pred ----------
    if args.process_pred:
        print('=== 处理 pred: pred * image ===')
        pred_path = join(data_dir, 'pred')
        pred_save_path = join(save_dir, 'pred_dte_map')
        process_map(pred_path, image_path, pred_save_path)
        print(f'pred 处理完成 → {pred_save_path}')

    # ---------- 处理 label ----------
    if args.process_label:
        print('=== 处理 label: label * image ===')
        label_path = join(data_dir, 'labelsTs')
        label_save_path = join(save_dir, 'label_dte_map')
        process_map(label_path, image_path, label_save_path)
        print(f'label 处理完成 → {label_save_path}')


if __name__ == '__main__':
    main()
