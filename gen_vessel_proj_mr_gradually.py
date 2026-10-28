### 把支架放到处理之后的MRI图像中，造影剂浓度从第0个角度逐渐增长到第150个角度

import numpy as np
import os, json, argparse
join = os.path.join
import nibabel as nib
from tqdm import tqdm
import tigre

from utils.file_rw import write_sp, read_sp
from utils.general import set_geo, coarse_mri_to_ct, add_skull_box_simple
from utils.marker import compose_stent_and_marker


def load_roi_mask_proj(proj_dir):
    proj = []
    for i in sorted(os.listdir(proj_dir)):
        proj.append(read_sp(join(proj_dir, i)))
    return np.stack(proj, axis=0)


def gen_roi_proj(image_path, label_path, vol_gt_mask_path, save_dir, split_json,
                 stent_path=None, marker_path=None, marker_mu_ratio=2.0):
    if marker_path is not None and stent_path is None:
        raise ValueError("marker_path requires stent_path")
    ct_vessel_image_save_path = join(save_dir, 'ct_vessel_image_2')
    cir_all_image_proj_save_dir = join(save_dir, 'image_proj', 'cir_all')
    rec_all_image_proj_save_dir = join(save_dir, 'image_proj', 'rec_all')
    rec_image_proj_save_dir = join(save_dir, 'image_proj', 'rec')
    cir_image_proj_save_dir = join(save_dir, 'image_proj', 'cir')
    cir_all_label_proj_save_dir = join(save_dir, 'label_proj', 'cir_all')
    rec_all_label_proj_save_dir = join(save_dir, 'label_proj', 'rec_all')
    rec_label_proj_save_dir = join(save_dir, 'label_proj', 'rec')
    cir_label_proj_save_dir = join(save_dir, 'label_proj', 'cir')
    os.makedirs(ct_vessel_image_save_path, exist_ok=True)

    ## 得到每个image文件对应的roi_mask文件的路径
    data_samples = dict()
    for i in os.listdir(vol_gt_mask_path):
        sample_name = i.split('_sample_')[0]
        if data_samples.get(sample_name) is None:
            data_samples[sample_name] = [i]
        else:
            data_samples[sample_name].append(i)

    stent_samples = None
    if stent_path is not None:
        stent_samples = dict()
        for i in os.listdir(stent_path):
            sample_name = i.split('_sample_')[0]
            key = i.replace('.nii.gz', '')
            if stent_samples.get(sample_name) is None:
                stent_samples[sample_name] = [key]
            else:
                stent_samples[sample_name].append(key)

    for phantom in tqdm(sorted(os.listdir(image_path))):
        if split_json is not None:
            if phantom.split('.nii.gz')[0] not in split_json.keys():
                print(f"{phantom} not in to be process, continue")
                continue

        ## 读取数据
        phantom_image_nib = nib.load(join(image_path, phantom))
        phantom_image_data = phantom_image_nib.get_fdata().astype(np.float32)
        phantom_image_data = coarse_mri_to_ct(phantom_image_data)
        phantom_image_data = add_skull_box_simple(phantom_image_data, skull_mu_min=0.01, skull_mu_max=0.015)

        phantom_label_nib = nib.load(join(label_path, phantom))
        phantom_label_data = np.round(phantom_label_nib.get_fdata()).astype(np.int16)
        ves_insert_value = np.random.uniform(0.08, 0.15)

        sample_points = data_samples.get(phantom.split('.nii.gz')[0])
        if sample_points is None:
            print(phantom, ' 在vol_gt_mask中不存在')
            continue

        ## 投影几何参数
        x_dim_max, y_dim_max, z_dim_max = phantom_image_nib.shape[0], phantom_image_nib.shape[1], phantom_image_nib.shape[2]
        [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in phantom_image_nib.header.get_zooms()[:3]]

        u_pixel_size, v_pixel_size = 0.308, 0.308
        u_detector, v_detector = 1500, 1500
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                nDetector=np.array([u_detector, v_detector]),
                dDetector=np.array([u_pixel_size, v_pixel_size]))
        angles = np.array(np.linspace(0, np.deg2rad(360), 361)[0:-1])

        ## 利用投影线性性质，分别投影各分量
        ## proj_bg: 背景（MRI转CT + 颅骨）
        proj_bg = tigre.Ax(np.transpose(phantom_image_data.astype(np.float32), [2, 0, 1]), geo, angles)

        ## proj_vessel_unit: 血管mask的单位响应（值为0/1），造影剂贡献 = w * ves_value * proj_vessel_unit
        proj_vessel_unit = tigre.Ax(np.transpose(phantom_label_data.astype(np.float32), [2, 0, 1]), geo, angles)

        for sample_point in sample_points:
            ## 支架分量（每个sample不同，需要单独投影）
            stent_vol = np.zeros_like(phantom_image_data, dtype=np.float32)
            marker_vol = np.zeros_like(phantom_image_data, dtype=np.float32)
            if stent_path is not None and stent_samples is not None:
                sample_stents = stent_samples.get(phantom.split('.nii.gz')[0])
                if sample_stents is not None and sample_point in sample_stents:
                    stent_nib = nib.load(join(stent_path, sample_point + '.nii.gz'))
                    stent_data = np.round(stent_nib.get_fdata()).astype(np.int8)
                    assert phantom_image_nib.shape == stent_data.shape
                    stent_insert_value = np.random.uniform(0.15, 0.25)
                    stent_vol[stent_data > 0] = stent_insert_value
                    if marker_path is not None:
                        marker_file = join(marker_path, sample_point + '.nii.gz')
                        if not os.path.exists(marker_file):
                            raise FileNotFoundError(
                                f"marker_path was provided but {marker_file} does not exist"
                            )
                        marker_nib = nib.load(marker_file)
                        marker_data = np.round(marker_nib.get_fdata()).astype(np.int8)
                        assert marker_nib.shape == stent_data.shape
                        marker_vol[marker_data > 0] = stent_insert_value * marker_mu_ratio
                else:
                    print(f'{phantom} {sample_point} not exist')
                    continue

            proj_stent = tigre.Ax(
                np.transpose(stent_vol + marker_vol, [2, 0, 1]), geo, angles
            )

            ## 合并3D体积保存（用于检查）
            vol_combined = phantom_image_data + stent_vol + marker_vol
            vol_combined[phantom_label_data > 0] = np.maximum(vol_combined[phantom_label_data > 0], ves_insert_value)
            nib.save(nib.Nifti1Image(vol_combined, phantom_image_nib.affine, phantom_image_nib.header),
                     join(ct_vessel_image_save_path, sample_point + '.nii.gz'))

            roi_proj = load_roi_mask_proj(join(vol_gt_mask_path, sample_point.split('.nii.gz')[0]))
            tar_num = 360
            tar_index = list(np.round(np.linspace(0, 360, tar_num + 1)[0:-1]).astype(np.int16))

            for i in tar_index:
                ## 造影剂权重：角度0→0，线性增长到角度150→1，150~360保持1
                if i <= 150:
                    w = i / 150.0
                else:
                    w = 1.0

                ## 按线性组合得到最终投影
                image_proj_slice = proj_bg[i] + proj_stent[i] + w * ves_insert_value * proj_vessel_unit[i]
                label_proj_slice = proj_vessel_unit[i]
                roi_proj_slice = roi_proj[i]

                coords = np.where(roi_proj_slice > 0)
                x_min, x_max, y_min, y_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])

                image_rec = image_proj_slice[x_min:x_max + 1, y_min:y_max + 1]
                label_rec = label_proj_slice[x_min:x_max + 1, y_min:y_max + 1]

                image_rec_all_mask = np.zeros_like(image_proj_slice, dtype=np.float32)
                image_rec_all_mask[x_min:x_max + 1, y_min:y_max + 1] = 1
                image_rec_all = image_proj_slice * image_rec_all_mask
                label_rec_all = label_proj_slice * image_rec_all_mask

                image_cir_all = image_proj_slice * roi_proj_slice
                label_cir_all = label_proj_slice * roi_proj_slice

                image_cir = image_cir_all[x_min:x_max + 1, y_min:y_max + 1]
                label_cir = label_cir_all[x_min:x_max + 1, y_min:y_max + 1]

                write_sp(image=image_cir_all, save_dir=join(cir_all_image_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_rec_all, save_dir=join(rec_all_image_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_cir, save_dir=join(cir_image_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}_0000.npz'))
                write_sp(image=image_rec, save_dir=join(rec_image_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}_0000.npz'))

                write_sp(image=label_cir_all, save_dir=join(cir_all_label_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_rec_all, save_dir=join(rec_all_label_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_cir, save_dir=join(cir_label_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}.npz'))
                write_sp(image=label_rec, save_dir=join(rec_label_proj_save_dir, sample_point, sample_point + f'_{str(int(i)).zfill(3)}.npz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="造影剂浓度从角度0到150逐渐增长，150~360保持不变")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--data_name", type=str, default='costa')
    parser.add_argument("--train", type=int, default=True)
    parser.add_argument("--test", type=int, default=True)
    parser.add_argument("--data_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--use_split", type=int, default=True)
    parser.add_argument("--marker_path", type=str, default=None)
    parser.add_argument("--marker_mu_ratio", type=float, default=2.0)
    parser.add_argument("--output_name", type=str, default="vessel_with_stent_gradually")
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    data_dir = args.data_dir
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    split = args.split
    use_split = args.use_split
    assert to_train in set([0, 1])
    assert to_test in set([0, 1])

    data_dir = join(data_dir, data_name)

    print(f'data_dir: {data_dir}\n'
          f'radius: {radius}\n',
          f'resolution: {resolution}\n')

    if to_train:
        print('生成训练数据 (造影剂渐变)')
        mode = 'train'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
        label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/{args.output_name}/radius-{radius}/{mode}_data')
        stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
        if use_split:
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path,
                     save_dir=save_dir, split_json=split_json, stent_path=stent_path,
                     marker_path=args.marker_path, marker_mu_ratio=args.marker_mu_ratio)

    if to_test:
        print('生成测试数据 (造影剂渐变)')
        mode = 'test'
        image_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/fill_hole_image')
        label_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/vessel/{mode}_data/label')
        vol_gt_mask_path = join(data_dir, f'voxel_{resolution}/new_proj/vol_gt_mask_proj/radius-{radius}/{mode}_data')
        save_dir = join(data_dir, f'voxel_{resolution}/new_proj/{args.output_name}/radius-{radius}/{mode}_data')
        stent_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/stent/{mode}_data/vol_gt')
        if use_split:
            json_path = join(data_dir, f'voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f"part_{split}.json")
            with open(json_path, "r", encoding="utf-8") as f:
                split_json = json.load(f)
        else:
            split_json = None
        gen_roi_proj(image_path=image_path, label_path=label_path, vol_gt_mask_path=vol_gt_mask_path,
                     save_dir=save_dir, split_json=split_json, stent_path=stent_path,
                     marker_path=args.marker_path, marker_mu_ratio=args.marker_mu_ratio)


# Historical launch command retained for reference; it is intentionally not
# executable Python code.  Use the shell command from DTE/gen_data/run.sh.
