#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import multiprocessing
import shutil
from time import sleep
from typing import Tuple

import SimpleITK
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm
from typing import Union

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.utilities.utils_gsl import get_dtm, calculate_class_percentage
 #计算dtm的类


class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        ##因为数据实际上会把不同通道的数据分开存放，所以读取数据的时候会将datas给vstack，也就是恢复为C, D, H ,W格式

        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        # print(data.shape)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:] ## 默认data是 (C, D, H, W)格式的。当然，会在读取时自动的加上C，原始数据就是正常的D, H, W格式的数据
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg) #返回的data和seg是裁剪之后的结果，也就是把需要裁剪的部分给赋值为-1
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:] ###这两句代码执行实际的resample操作
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)

        # print(np.unique(seg))

        ## 应该要在这里计算数据的DTM
        # label_list = list(dataset_json['labels'].values())
        label_list = [0, 1] #我直接把它赋值了
        dtm = np.zeros_like(np.stack([data] * len(label_list), axis=-1), dtype=np.float32) # (Channel, D, H, W, Class)
        seg_to_compute_dtm = np.copy(seg)
        seg_to_compute_dtm[seg_to_compute_dtm==-1]=0 #把-1替换为0
        for data_slice_index in range(data.shape[0]): #对每个通道分别计算dtm
            dtm[data_slice_index] = get_dtm(mask=seg_to_compute_dtm[data_slice_index], label_list=label_list, voxel_spacing=target_spacing)
            # print(target_spacing)

        # print('dataset.json: ', dataset_json)
        # print(label_list)
        # print(type((label_list[0])))

        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

        ## 也要顺便计算当前处理的批次中的数据中的各类的百分比占比
        # print(np.all(seg > 0))
        class_percentage_result = calculate_class_percentage(seg_to_compute_dtm)
        properties['class_percentage_result'] = class_percentage_result

        return data, seg, dtm, properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if self.verbose:
            print(seg_file)

        ## 这里的data_properties是同一例数据的一些属性，包括spacing和affine。
        # print('==========================')
        # print(data_properties)
        data, seg, dtm, data_properties = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        return data, seg, dtm, data_properties

    #     ## new add by ly 2025-12-02
    # def run_case_save_with_result(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
    #                              plans_manager: PlansManager, configuration_manager: ConfigurationManager,
    #                              dataset_json: Union[dict, str], case_key: str, shared_results: multiprocessing.Manager.dict):
    #     """
    #     运行 run_case_save 并将 case 的类百分比结果存入共享字典。
    #     """
    #     data, seg, dtm, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        
    #     # **获取计算得到的类百分比**
    #     class_percentage_result = properties.get('class_percentage_result', {})

    #     # ... (run_case_save 中原有的保存数据到磁盘的代码，保持不变) ...
    #     data = data.astype(np.float32, copy=False)
    #     seg = seg.astype(np.int16, copy=False)
    #     dtm = dtm.astype(np.float32, copy=False)
    #     # print('dtm shape: ', dtm.shape)
    #     # print(configuration_manager.patch_size)
        
    #     block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
    #         data.shape,
    #         tuple(configuration_manager.patch_size),
    #         data.itemsize)
    #     block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(
    #         seg.shape,
    #         tuple(configuration_manager.patch_size),
    #         seg.itemsize)
        
    #     # 将5D的dtm分为多个4D来计算，也就是对第5个维度（class）进行拆分
    #     block_size_dtm, chunk_size_dtm = nnUNetDatasetBlosc2.comp_blosc2_params(
    #         dtm[...,0].shape,
    #         tuple(configuration_manager.patch_size),
    #         dtm[...,0].itemsize)
    #     # print('==============================')
    #     # print('data.shape, seg.shape, dtm.shape  ', data.shape, seg.shape, dtm.shape)
        
    #     nnUNetDatasetBlosc2.save_case_ly(data, seg, dtm, properties, output_filename_truncated,
    #                                      chunks=chunk_size_data, blocks=block_size_data,
    #                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg,
    #                                      chunks_dtm=chunk_size_dtm, blocks_dtm=block_size_dtm)
        
    #     # **将结果存储到共享字典中**
    #     shared_results[case_key] = class_percentage_result

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, dtm, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print('==========================')
        # print(plans_manager)
        # print('==========================')
        # print(configuration_manager)
        # print('==========================')
        # print(dataset_json)

        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16, copy=False)
        dtm = dtm.astype(np.float32, copy=False)
        # print('dtm shape: ', dtm.shape)
        # print(configuration_manager.patch_size)
        # print('dtypes', data.dtype, seg.dtype)
        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
            data.shape,
            tuple(configuration_manager.patch_size),
            data.itemsize)
        block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(
            seg.shape,
            tuple(configuration_manager.patch_size),
            seg.itemsize)
        
        # 将5D的dtm分为多个4D来计算，也就是对第5个维度（class）进行拆分
        block_size_dtm, chunk_size_dtm = nnUNetDatasetBlosc2.comp_blosc2_params(
            dtm[...,0].shape,
            tuple(configuration_manager.patch_size),
            dtm[...,0].itemsize)
        # print('==============================')
        # print('data.shape, seg.shape, dtm.shape  ', data.shape, seg.shape, dtm.shape)
        # print('chunk size data: ',chunk_size_data)
        # print('block_size_data: ',block_size_data)
        # print('chunk_size_seg: ',chunk_size_seg)
        # print('block_size_seg: ',block_size_seg)
        # print('chunk_size_dtm: ',chunk_size_dtm)
        # print('block_size_dtm: ',block_size_dtm)
        ### chunk: 磁盘IO的基本单元. block: cpu进行压缩的基本单元。这里对训练数据和标签赋予不同的chunk和block大小。但是实际上应该是差不多的
        # print(output_filename_truncated, properties)
        ## 下面函数是使数据的压缩率最高，同时IO速度最快
        nnUNetDatasetBlosc2.save_case_ly(data, seg, dtm, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg,
                                      chunks_dtm=chunk_size_dtm, blocks_dtm=block_size_dtm)

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        foreground_mask = seg != 0
        foreground_coords = np.argwhere(foreground_mask)
        seg = seg[foreground_mask]
        del foreground_mask
        unique_labels = pd.unique(seg.ravel())

        # We don't need more than 1e7 foreground samples. That's insanity. Cap here
        if len(foreground_coords) > 1e7:
            take_every = math.floor(len(foreground_coords) / 1e7)
            # keep computation time reasonable
            if verbose:
                print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
            foreground_coords = foreground_coords[::take_every]
            seg = seg[::take_every]

        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)

            # check if any of the labels are in seg, if not skip c
            if isinstance(c, (tuple, list)):
                if not any([ci in unique_labels for ci in c]):
                    class_locs[k] = []
                    continue
            else:
                if c not in unique_labels:
                    class_locs[k] = []
                    continue

            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = foreground_coords[mask]
            else:
                mask = seg == c
                all_locs = foreground_coords[mask]
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
            seg = seg[~mask]
            foreground_coords = foreground_coords[~mask]
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]


        # 共享字典用于存储每个 case 的百分比结果
        # manager = multiprocessing.Manager()
        # class_percentage_results = manager.dict()

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            # print(dataset)
            for k in dataset.keys(): #这里是循环遍历依次读取并处理
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

                ## 试图获取全局的类别百分占比
                # r.append(p.starmap_async(self.run_case_save,
                #             ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                #             plans_manager, configuration_manager,
                #             dataset_json, k, class_percentage_results),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

            ## 所有进程都计算完成之后，计算全部的类别百分比
            # res = self._calculate_global_percentage(class_percentage_results, dataset_json)
        # print(res)

    # ## new add by ly 2025-12-02
    # def _calculate_global_percentage(self, class_percentage_results: dict, dataset_json: dict):
    #     """
    #     计算全局类的总百分比。
    #     """
    #     # 获取所有类别的标签
    #     all_labels = list(dataset_json['labels'].values())
        
    #     # 初始化全局百分比累加器，键为标签，值为总和
    #     global_percentage = {label: 0.0 for label in all_labels}
        
    #     print("\n--- Class Percentage Results (Per Case) ---")
    #     for case_key, result in class_percentage_results.items():
    #         print(f"Case {case_key}: {result}")
    #         # 累加每个 case 的百分比
    #         for label, percentage in result.items():
    #              # 确保只累加所有 case 中存在的类（如果您的 calculate_class_percentage 函数只返回前景类）
    #             if label in global_percentage:
    #                 global_percentage[label] += percentage
        
    #     # 结果是所有数据的总和，不是一个真正的“百分比”，因为它是每个数据百分比的直接累加。
    #     # 如果需要的是所有像素的**全局平均百分比**，则需要将每个 case 的总像素数也存储起来。
    #     # 假设您只需要每个 case 百分比的**直接总和**作为“全局百分比”：
    #     print("\n--- Global Class Percentage Sum (Sum of all individual case percentages) ---")
    #     # print(global_percentage)
    #     return global_percentage
    

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    # example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
    seg = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage('/home/isensee/temp/H-mito-val-v2.nii.gz'))[None]
    DefaultPreprocessor._sample_foreground_locations(seg, np.arange(1, np.max(seg) + 1))



