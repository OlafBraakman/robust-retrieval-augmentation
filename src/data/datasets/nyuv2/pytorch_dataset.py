# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from ..dataset_base import DatasetBaseDepth
from .nyuv2 import NYUv2Base

class NYUv2(NYUv2Base, DatasetBaseDepth):

    def __init__(self,
                 data_dir=None,
                 split='train',
                 depth_mode='refined'):
        super(NYUv2, self).__init__()
        
        assert split in self.SPLITS
        # assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._split = split
        self._depth_mode = depth_mode

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            self.img_dir, self.depth_dir, self.label_dir = \
                self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return 17.206409554343555
    
    @property
    def depth_std(self):
        return 8.610794655333109

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    def get_image_path(self, idx):
        img_dir = self.img_dir[self._split]
        return os.path.join(self._data_dir, img_dir[idx])

    def get_depth_path(self, idx):
        depth_dir = self.depth_dir[self._split]
        depth_file = depth_dir[idx]
        return os.path.join(self._data_dir, depth_file)
    
    def load_image(self, idx):
        img_dir = self.img_dir[self._split]
        fp = os.path.join(self._data_dir, img_dir[idx])
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # Based on values from https://github.com/facebookresearch/omnivore/issues/12#issuecomment-1070911016
    def get_baseline(self) -> float:
        return 0.075 # only kv1 in this dataset
       
    def get_focal_length(self) -> float:
        return 518.86 # Based on average focal length for kv1 cameras in SUNRGBD

    def load_depth(self, idx):
        depth_dir = self.depth_dir[self._split]
        depth_file = depth_dir[idx]

        fp = os.path.join(self._data_dir, depth_file)
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

        focal_length = self.get_focal_length()
        baseline = self.get_baseline()

        # return depth
        depth = depth.astype(np.float32) / 1000.0
        depth[depth > 10] = 10

        disparity = baseline * focal_length / depth
        return disparity
    
    def load_label(self, idx) -> int:
        label_dir = self.label_dir[self._split]
        with open(os.path.join(self._data_dir, label_dir[idx]), 'r') as f:
            label = f.readline()

        return self.SCENE_MAP[self.map_scene(label)]

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in self.SPLITS:
            file = _get_filepath(f'{phase}.txt')

            img_dir[phase] = self.list_from_file(file, phase, "rgb")
            depth_dir[phase] = self.list_from_file(file, phase, "depth_raw" if self._depth_mode == "raw" else "depth")
            label_dir[phase] = self.list_from_file(file, phase, "scene")

        return img_dir, depth_dir, label_dir
    
    def list_from_file(self, filepath, split, type):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()

        file_name = lambda id: str(id).zfill(4) + ".txt" if type == "scene" else str(id).zfill(4) + ".png"
        file_list = [os.path.join(split, type, file_name(id)) for id in file_list]
       
        return file_list
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        label_dir = self.label_dir[self._split]
       
        with open(os.path.join(self._data_dir, label_dir[idx]), 'r') as f:
            label = f.readline()

        item[0]['raw_label'] = label
        return item

    def __len__(self):
        return len(self.img_dir[self._split])
        