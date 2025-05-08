from .imagenet import ImageNetBase
from ..dataset_base import DatasetBase

import os
from PIL import Image
import numpy as np

class ImageNet64(ImageNetBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train'):
        super(ImageNet64, self).__init__()

        assert split in self.SPLITS, \
            f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            self.img_dir, self.label_dir = \
                self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

    @property
    def split(self) -> str:
        return self._split

    @property
    def source_path(self) -> str:
        return os.path.abspath(os.path.dirname(__file__))

    def get_image_path(self, idx):
        img_dir = self.img_dir[self._split]
        return os.path.join(self._data_dir, img_dir[idx])
    
    def load_image(self, idx):
        img_dir = self.img_dir[self._split]
        fp = os.path.join(self._data_dir, img_dir[idx])

        image = Image.open(fp).convert("RGB")
        image = np.array(image)
        return image

    def load_label(self, idx) -> int:
        label = self.label_dir[self._split][idx]
        return int(label) - 1

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir = dict()
        label_dir = dict()

        for phase in self.SPLITS:
            rgb_file = _get_filepath(f'classification_{phase}_rgb.txt')
            img_dir[phase] = self.list_from_file(rgb_file)

            label_file = _get_filepath(f'classification_{phase}_label.txt')
            label_dir[phase] = self.list_from_file(label_file)

        return img_dir, label_dir
    
    def list_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
       
        return file_list

    def __len__(self):
        return len(self.img_dir[self._split])