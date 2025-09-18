import copy
import pdb

import mmengine
import os.path as osp
import mmengine.fileio as fileio
from .basesegdataset import BaseSegDataset
from mmengine.dataset import BaseDataset, Compose
from mmseg.registry import DATASETS
from typing import Callable, Dict, List, Optional, Sequence, Union


@DATASETS.register_module()
class MedicalSegDataset(BaseSegDataset):
    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(
            copy.deepcopy(metainfo))  # classes': ('background', 'foreground'), 'palette': [[0, 0, 0], [255, 255, 255]]

        # Get label map for custom classes
        self.label_map = self._metainfo.get('label_map', None)
        self._metainfo.update(
            dict(
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    def _get_datainfo_from_file(self, anno_file):
        data_list = list()
        if osp.isfile(anno_file):
            lines = mmengine.list_from_file(
                anno_file, backend_args=self.backend_args)
            for line in lines:
                data_info = dict()
                line = line.strip()
                if ' /' in line:
                    items = line.split(' /')
                    img_path = items[0].strip()
                    seg_map_path = '/' + items[1].strip()
                else:
                    img_path = line
                    seg_map_path = ""
                data_info["img_path"] = img_path
                data_info["seg_map_path"] = seg_map_path
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        return data_list

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """

        data_list = list()
        if (type(self.ann_file) is list):
            for sub_ann_file in self.ann_file:
                sub_data_list = self._get_datainfo_from_file(sub_ann_file)
                data_list.extend(sub_data_list)
        elif (osp.isfile(self.ann_file)):
            data_list = self._get_datainfo_from_file(self.ann_file)

        return data_list
