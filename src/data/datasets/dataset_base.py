# From: https://github.com/Barchid/RGBD-Seg/blob/master/dataloaders/dataset_base.py
import os
import pickle
import abc

import numpy as np
from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Any

KeyType = SimpleNamespace(
    IMAGE="image",
    DEPTH="depth",
    LABEL="label",
)

class DatasetBase(abc.ABC, Dataset):
    def __init__(self):
        self._camera = None
        self._default_preprocessor = lambda x: x
        self.preprocessor = self._default_preprocessor

    def __enter__(self):
        return self

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        image = self.load_image(idx)  
        label = self.load_label(idx)
        
        image = self.preprocessor(image)

        return (image, label), idx
        # sample = {
        #     'image': self.load_image(idx)                  
        # }
        # sample = self.preprocessor(sample)
        # sample['label'] = label

        # return sample, idx

    @property
    @abc.abstractmethod
    def split(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def source_path(self) -> str:
        pass

    @abc.abstractmethod
    def load_image(self, idx) -> Any:
        pass

    @abc.abstractmethod
    def get_image_path(self, idx) -> str:
        pass

    @abc.abstractmethod
    def load_label(self, idx) -> Any:
        pass
    
class DatasetBaseDepth(DatasetBase):
    def __init__(self, modality="image"):
        self._camera = None
        self._default_preprocessor = lambda x: x
        self.preprocessor = self._default_preprocessor
        self.modality = modality

    def __getitem__(self, idx):
        image = self.load_image(idx) if self.modality == "image" else self.load_depth(idx)  
        label = self.load_label(idx)
        
        image = self.preprocessor(image)

        return (image, label), idx
        # sample = {
        #     'image': self.load_image(idx),
        #     'depth': self.load_depth(idx)
        # }
        # sample = self.preprocessor(sample)
        # sample['label'] = label

        # return sample, idx

    @property
    @abc.abstractmethod
    def depth_mode(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def depth_mean(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def depth_std(self) -> float:
        pass

    @abc.abstractmethod
    def load_depth(self, idx) -> Any:
        pass

    @abc.abstractmethod
    def get_depth_path(self, idx) -> str:
        pass

    def compute_depth_mean_std(self, force_recompute=False):
        # ensure that mean and std are computed on train set only
        assert self.split == 'train'

        # build filename
        depth_stats_filepath = os.path.join(
            self.source_path, f'depth_{self.depth_mode}_mean_std.pickle')

        if not force_recompute and os.path.exists(depth_stats_filepath):
            depth_stats = pickle.load(open(depth_stats_filepath, 'rb'))
            print(f'Loaded depth mean and std from {depth_stats_filepath}')
            print(depth_stats)
            return depth_stats

        print('Compute mean and std for depth images.')

        pixel_sum = np.float64(0)
        pixel_nr = np.uint64(0)
        std_sum = np.float64(0)

        min_depth = np.inf
        max_depth = -np.inf

        print('Compute mean')
        for i in range(len(self)):
            depth = self.load_depth(i)
            min_depth = min(depth.min(), min_depth)
            max_depth = max(depth.max(), max_depth)
            if self.depth_mode == 'raw':
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()
            pixel_sum += np.sum(depth_valid)
            pixel_nr += np.uint64(len(depth_valid))
            print(f'\r{i+1}/{len(self)}', end='')
        print()

        print(min_depth)
        print(max_depth)

        mean = pixel_sum / pixel_nr

        print('Compute std')
        for i in range(len(self)):
            depth = self.load_depth(i)
            if self.depth_mode == 'raw':
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()
            std_sum += np.sum(np.square(depth_valid - mean))
            print(f'\r{i+1}/{len(self)}', end='')
        print()

        std = np.sqrt(std_sum / pixel_nr)

        depth_stats = {'mean': mean, 'std': std}
        print(depth_stats)

        with open(depth_stats_filepath, 'wb') as f:
            pickle.dump(depth_stats, f)

        return depth_stats
