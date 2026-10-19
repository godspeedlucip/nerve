from typing import Union, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset_dtm import nnUNetDatasetBlosc2DTM
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils_gsl import get_dtm, calculate_class_percentage
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

import multiprocessing
import shutil
from time import sleep
from tqdm import tqdm


class DefaultPreprocessorDTM(DefaultPreprocessor):
    """Preprocessor that additionally computes and stores Distance Transform Maps (DTM)."""

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        data = data.astype(np.float32)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation."
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)

        # Compute DTM
        label_list = [0, 1]
        dtm = np.zeros_like(np.stack([data] * len(label_list), axis=-1), dtype=np.float32)
        seg_to_compute_dtm = np.copy(seg)
        seg_to_compute_dtm[seg_to_compute_dtm == -1] = 0
        for data_slice_index in range(data.shape[0]):
            dtm[data_slice_index] = get_dtm(mask=seg_to_compute_dtm[data_slice_index],
                                            label_list=label_list,
                                            voxel_spacing=target_spacing)

        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                              verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)

        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

        class_percentage_result = calculate_class_percentage(seg_to_compute_dtm)
        properties['class_percentage_result'] = class_percentage_result

        return data, seg, dtm, properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        data, data_properties = rw.read_images(image_files)
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if self.verbose:
            print(seg_file)

        data, seg, dtm, data_properties = self.run_case_npy(data, seg, data_properties,
                                                             plans_manager, configuration_manager,
                                                             dataset_json)
        return data, seg, dtm, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, dtm, properties = self.run_case(image_files, seg_file, plans_manager,
                                                    configuration_manager, dataset_json)

        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16, copy=False)
        dtm = dtm.astype(np.float32, copy=False)

        block_size_data, chunk_size_data = nnUNetDatasetBlosc2DTM.comp_blosc2_params(
            data.shape, tuple(configuration_manager.patch_size), data.itemsize)
        block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2DTM.comp_blosc2_params(
            seg.shape, tuple(configuration_manager.patch_size), seg.itemsize)
        block_size_dtm, chunk_size_dtm = nnUNetDatasetBlosc2DTM.comp_blosc2_params(
            dtm[..., 0].shape, tuple(configuration_manager.patch_size), dtm[..., 0].itemsize)

        nnUNetDatasetBlosc2DTM.save_case(data, seg, dtm, properties, output_filename_truncated,
                                          chunks=chunk_size_data, blocks=block_size_data,
                                          chunks_seg=chunk_size_seg, blocks_seg=block_size_seg,
                                          chunks_dtm=chunk_size_dtm, blocks_dtm=block_size_dtm)

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, maybe_mkdir_p, join, load_json

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), f"Expected plans file ({plans_file}) not found."
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            workers = [j for j in p._pool]
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                    done = [i for i in remaining if r[i].ready()]
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
