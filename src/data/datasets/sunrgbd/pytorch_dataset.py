# -*- coding: utf-8 -*-
# From: https://github.com/Barchid/RGBD-Seg/blob/master/dataloaders/sunrgbd/pytorch_dataset.py
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2

from .sunrgbd import SUNRBDBase
from ..dataset_base import DatasetBaseDepth, DatasetBase
from pathlib import Path


class SUNRGBD(SUNRBDBase, DatasetBaseDepth):
    def __init__(self,
                 data_dir=None,
                 split='train',
                 depth_mode='refined', modality="image"):
        super(SUNRGBD, self).__init__(modality=modality)

        assert split in self.SPLITS, \
            f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split

        assert depth_mode in ['refined', 'raw']
        self._depth_mode = depth_mode

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self.img_dir, self.depth_dir, self.label_dir = \
                self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    @property
    def depth_mean(self):
        return 24.82

    @property
    def depth_std(self):
        return 14.40

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    def load_image(self, idx):
        img_dir = self.img_dir[self._split]
        fp = os.path.join(self._data_dir, img_dir[idx])
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_image_path(self, idx):
        img_dir = self.img_dir[self._split]
        return os.path.join(self._data_dir, img_dir[idx])

    # Based on values from https://github.com/facebookresearch/omnivore/issues/12#issuecomment-1070911016
    def get_baseline(self, path: str) -> float:
        if "kv1" in path:
            return 0.075
        elif "kv2" in path:
            return 0.075
        elif "realsense" in path:
            return 0.095
        elif "xtion" in path:
            return 0.095 # guessed based on length of 18cm for ASUS xtion v1
        else:
            raise Exception(f"No baseline found for path: {path}")

    def load_depth(self, idx):
        depth_dir = self.depth_dir[self._split]
       
        if self._depth_mode == 'raw':
            depth_file = depth_dir[idx].replace('depth_bfx', 'depth')
        else:
            depth_file = depth_dir[idx]

        fp = os.path.join(self._data_dir, depth_file)
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

        focal_path = Path(fp).parents[1] / "intrinsics.txt"
        focal_length = float(focal_path.read_text().strip().split()[0])

        # Load baseline for the depth sensor based on path file name
        baseline = self.get_baseline(depth_file)

        depth = ((depth >> 3) | (depth << 13)).astype(np.float32) / 1000.0
        depth[depth > 8] = 8

        disparity = baseline * focal_length / depth
        return disparity
    
        # Disparity calculation for Omnivore

        # depth = np.array(depth).astype(np.float32)
        # depth_in_meters = depth / 1000.
        # # if min_depth is not None:
        # depth_in_meters = depth_in_meters.clip(min=0.01, max=50)
        # disparity = baseline * focal_length / depth_in_meters
        # return disparity
    
    def get_depth_path(self, idx):
        depth_dir = self.depth_dir[self._split]
       
        if self._depth_mode == 'raw':
            depth_file = depth_dir[idx].replace('depth_bfx', 'depth')
        else:
            depth_file = depth_dir[idx]
        return os.path.join(self._data_dir, depth_file)
    
    def load_label(self, idx) -> int:
        label_dir = self.label_dir[self._split]
        
        with open(os.path.join(self._data_dir, label_dir[idx]), 'r') as f:
            label = f.readline()

        return self.SCENE_MAP[label]

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in ['train', 'test']:
            img_dir_file = _get_filepath(f'classification_{phase}_rgb.txt')
            depth_dir_file = _get_filepath(f'classification_{phase}_depth.txt')
            label_dir_file = _get_filepath(f'classification_{phase}_scene.txt')

            img_dir[phase] = self.list_from_file(img_dir_file)
            depth_dir[phase] = self.list_from_file(depth_dir_file)
            label_dir[phase] = self.list_from_file(label_dir_file)

        return img_dir, depth_dir, label_dir

    def list_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        
        return file_list

    def __len__(self):
        return len(self.img_dir[self._split])

class SUNRGB(SUNRBDBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train'):
        super(SUNRGB, self).__init__()

        assert split in self.SPLITS, \
            f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split


        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self.img_dir, self.label_dir = \
                self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

    @property
    def split(self):
        return self._split

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    def load_image(self, idx):
        img_dir = self.img_dir[self._split]
        fp = os.path.join(self._data_dir, img_dir[idx])
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_image_path(self, idx):
        img_dir = self.img_dir[self._split]
        return os.path.join(self._data_dir, img_dir[idx])

    def load_label(self, idx) -> int:
        label_dir = self.label_dir[self._split]
        
        with open(os.path.join(self._data_dir, label_dir[idx]), 'r') as f:
            label = f.readline()

        return self.SCENE_MAP[label]

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir = dict()
        label_dir = dict()

        for phase in ['train', 'test']:
            img_dir_file = _get_filepath(f'classification_{phase}_rgb.txt')
            label_dir_file = _get_filepath(f'classification_{phase}_scene.txt')

            img_dir[phase] = self.list_from_file(img_dir_file)
            label_dir[phase] = self.list_from_file(label_dir_file)

        return img_dir, label_dir

    def list_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        
        return file_list

    def __len__(self):
        return len(self.img_dir[self._split])
