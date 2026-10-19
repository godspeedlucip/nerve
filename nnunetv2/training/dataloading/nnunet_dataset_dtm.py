import os
from typing import List

import numpy as np
import blosc2
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, write_pickle

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


class nnUNetDatasetBlosc2DTM(nnUNetDatasetBlosc2):
    """DTM-aware Blosc2 dataset. Loads distance transform maps alongside data and seg."""

    def load_case(self, identifier):
        dparams = {'nthreads': 1}
        mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        valid_class = list(properties['class_locations'].keys())
        valid_class = list(set(valid_class).union({0}))

        dtm_list = []
        for i in valid_class:
            dtm_b2nd_file = join(self.source_folder, identifier + f'_dtm_class{str(i)}.b2nd')
            dtm = blosc2.open(urlpath=dtm_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
            dtm_list.append(dtm)
        dtm = np.stack(dtm_list, axis=-1)

        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')
        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_b2nd_file = join(self.folder_with_segs_from_previous_stage, identifier + '.b2nd')
            seg_prev = blosc2.open(urlpath=prev_seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        else:
            seg_prev = None

        return data, seg, dtm, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            dtm: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            chunks_dtm=None,
            blocks_dtm=None,
            clevel: int = 8,
            codec=blosc2.Codec.ZSTD
    ):
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks
        if blocks_dtm is None:
            blocks_dtm = blocks

        cparams = {
            'codec': codec,
            'clevel': clevel,
        }

        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + '.b2nd',
                       chunks=chunks, blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + '_seg.b2nd',
                       chunks=chunks_seg, blocks=blocks_seg, cparams=cparams)

        for i in range(dtm.shape[-1]):
            blosc2.asarray(np.ascontiguousarray(dtm[..., i]),
                           urlpath=output_filename_truncated + f'_dtm_class{i}.b2nd',
                           chunks=chunks_dtm, blocks=blocks_dtm, cparams=cparams)

        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        case_identifiers = [i[:-5] for i in os.listdir(folder)
                           if i.endswith(".b2nd")
                           and not i.endswith("_seg.b2nd")
                           and i.find("_dtm_class") == -1]
        return case_identifiers
