import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoaderDTM(DataLoader):
    """DTM-aware DataLoader. Like nnUNetDataLoader but also loads, crops, pads, and transforms DTM."""

    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        self.indices = data.identifiers
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape, self.dtm_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        data, seg, dtm, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        channels_dtm = dtm.shape[0]
        class_dtm = dtm.shape[-1]
        dtm_shape = (self.batch_size, channels_dtm, *self.patch_size, class_dtm)
        return data_shape, seg_shape, dtm_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys()
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False
                       for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])
                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                else:
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        dtm_all = np.zeros(self.dtm_shape, dtype=np.float32)

        dtm_class0_all = np.zeros(self.data_shape, dtype=np.float32)
        dtm_class1_all = np.zeros(self.data_shape, dtype=np.float32)
        valid_mask_all = np.zeros(self.data_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, dtm, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            valid_mask_all[j] = crop_and_pad_nd(np.ones_like(seg, dtype=np.int16), bbox, 0)
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            dtm_class0_all[j] = crop_and_pad_nd(dtm[..., 0], bbox, 0)
            dtm_class1_all[j] = crop_and_pad_nd(dtm[..., 1], bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    dtm_class0_all = torch.from_numpy(dtm_class0_all)
                    dtm_class1_all = torch.from_numpy(dtm_class1_all)
                    valid_mask_all = torch.from_numpy(valid_mask_all.astype(np.int16)).to(torch.int16)

                    images, segs, dtms_class0, dtms_class1, valid_masks = [], [], [], [], []

                    for b in range(self.batch_size):
                        tmp = self.transforms(**{
                            'image': data_all[b],
                            'segmentation': seg_all[b],
                            'dtm_class0': dtm_class0_all[b],
                            'dtm_class1': dtm_class1_all[b],
                            'valid_mask': valid_mask_all[b]
                        })
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        dtms_class0.append(tmp['dtm_class0'])
                        dtms_class1.append(tmp['dtm_class1'])
                        valid_masks.append(tmp['valid_mask'])

                    data_all = torch.stack(images)

                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)

                    dtm_all = None
                    if isinstance(dtms_class0[0], list):
                        dtm_class0_stacked = [torch.stack([s[i] for s in dtms_class0]) for i in range(len(dtms_class0[0]))]
                        dtm_class1_stacked = [torch.stack([s[i] for s in dtms_class1]) for i in range(len(dtms_class1[0]))]
                        dtm_all = [torch.stack((dtm_class0_stacked[i], dtm_class1_stacked[i]), dim=-1)
                                   for i in range(len(dtm_class0_stacked))]
                    else:
                        dtm_class0_stacked = torch.stack(dtms_class0)
                        dtm_class1_stacked = torch.stack(dtms_class1)
                        dtm_all = torch.stack((dtm_class0_stacked, dtm_class1_stacked), dim=-1)

                    if isinstance(valid_masks[0], list):
                        valid_mask_all = [torch.stack([s[i] for s in valid_masks]) for i in range(len(valid_masks[0]))]
                    else:
                        valid_mask_all = torch.stack(valid_masks)

                    del segs, images, dtms_class0, dtms_class1, valid_masks
                    return {'data': data_all, 'target': seg_all, 'keys': selected_keys,
                            'dtm': dtm_all, 'valid_mask': valid_mask_all}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys,
                'dtm': None, 'valid_mask': None}
