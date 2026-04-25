# Copyright (c) 2025 Caries-DETR Authors. All rights reserved.
"""AlphaDent dataset class for Caries-DETR.

The AlphaDent dataset contains intraoral photographs annotated with nine
object categories covering common dental pathologies and restorations.
The original dataset uses the Ultralytics/YOLO instance segmentation format
(.jpg images + .txt label files + yolo_seg_train.yaml); annotations should
be converted to COCO JSON format before use with this loader.
Annotations follow the standard COCO JSON schema.
"""

import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class AlphaDentDataset(BaseDetDataset):
    """AlphaDent intraoral photograph dataset (9 classes).

    The dataset is distributed in Ultralytics/YOLO instance segmentation
    format and must be converted to COCO JSON format prior to use.

    Categories:
        0 - Abrasion
        1 - Filling
        2 - Crown
        3 - Caries 1 class  (early-stage caries)
        4 - Caries 2 class
        5 - Caries 3 class
        6 - Caries 4 class
        7 - Caries 5 class
        8 - Caries 6 class  (severe caries / pulp involvement)

    The annotation format follows the COCO JSON schema.
    """

    METAINFO = {
        'classes': (
            'Abrasion',
            'Filling',
            'Crown',
            'Caries 1 class',
            'Caries 2 class',
            'Caries 3 class',
            'Caries 4 class',
            'Caries 5 class',
            'Caries 6 class',
        ),
        'palette': [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
        ],
    }

    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from the COCO-format annotation file."""
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []

        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info,
            })
            data_list.append(parsed)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), (
                f"Annotation ids in '{self.ann_file}' are not unique!"
            )

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to the target format.

        Args:
            raw_data_info (dict): Raw data information loaded from the
                annotation file.

        Returns:
            dict: Parsed annotation dictionary.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])

        if self.data_prefix.get('seg'):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None

        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            instance = {}
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            instance['ignore_flag'] = 1 if ann.get('iscrowd', False) else 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            if ann.get('segmentation'):
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list
        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = {d['img_id'] for d in self.data_list}
        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann

        valid = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(data_info['width'], data_info['height']) >= min_size:
                valid.append(data_info)
        return valid
