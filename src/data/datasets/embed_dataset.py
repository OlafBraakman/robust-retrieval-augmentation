from torch.utils.data import Dataset
from .dataset_base import DatasetBase
from typing import Literal
from tools.registry import Registry
from .embedding import Embedding, paired_random, single_random
from tqdm import tqdm
from data.datasets.build import build_dataset
from data.datasets.dataset_base import DatasetBaseDepth


embed_dataset_registry = Registry()

class EmbedStrategy:
    num_augmentations = 25

class EmbedDataset(Dataset):

    @classmethod
    def from_config(cls, dataset_config):

        dataset = build_dataset(dataset_config)
        
        # tag_map = {
        #     'image': dataset_config['image_tag'],
        #     'depth': dataset_config['depth_tag']
        # }

        return cls(dataset, dataset_config['image_tag'])

    # def __init__(self, dataset: DatasetBase, tag_map: dict[Literal['image', 'depth'], tuple[str, str, str]], in_memory=True, use_augmentations=True, normalized=True):

    def __init__(self, dataset: DatasetBase, tag, in_memory=True, use_augmentations=True, normalized=True):
        self.dataset = dataset
        self.use_augmentations = use_augmentations

        self.image_mem = None
        self.depth_mem = None

        self.normalized = normalized

        self.in_memory = in_memory
        self.tag = tag

        if self.in_memory:

            if tag[0] == "image":
                self.image_mem = [Embedding.from_dataset(self.get_path(idx, 'image'), *tag) for idx in tqdm(range(len(dataset)))]
            elif tag[0] == "depth" and self.is_rgbd():
                self.image_mem = [Embedding.from_dataset(self.get_path(idx, 'depth'), *tag) for idx in tqdm(range(len(dataset)))]
            else:
                raise Exception("Not a valid embed dataset configuration")

    def is_rgb(self):
        return isinstance(self.dataset, DatasetBase)

    def is_rgbd(self):
        return isinstance(self.dataset, DatasetBaseDepth)

    def __len__(self):
        return self.dataset.__len__()

    def get_path(self, idx, type: Literal['image', 'depth']):
        if type == "image":
            return self.dataset.get_image_path(idx)
        elif type == "depth":
            return self.dataset.get_depth_path(idx)

    def __getitem__(self, idx):
        label = self.dataset.load_label(idx)
        image_embedding = self.image_mem[idx] if self.in_memory else Embedding.from_dataset(self.get_path(idx, self.tag[0]), *self.tag)
        embedding = image_embedding.embedding / image_embedding.embedding.norm(dim=-1, keepdim=True)
        return (embedding, label), idx

class AugmentationSampler(Dataset):

    def __init__(self, num_augmentations, dataset, index):
        self.num_augmentations = num_augmentations
        self.dataset = dataset
        self.index = index

    def __len__(self):
        return self.num_augmentations

    def __getitem__(self, _):
        return self.dataset[self.index]


def build_embed_dataset(embed_dataset_cfg):
    retrieval_augmentation_module = embed_dataset_registry[embed_dataset_cfg['name']](embed_dataset_cfg)
    return retrieval_augmentation_module
