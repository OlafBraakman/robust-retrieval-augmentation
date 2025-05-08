import torch

from .embedding import Embedding
from data.datasets.build import build_dataset

from typing import Literal
from .dataset_base import DatasetBase, DatasetBaseDepth

from tqdm import tqdm

class EmbedMemory():

    # supported_modalities = ['image', 'depth']

    @classmethod
    def from_config(cls, dataset_config, memory_config):
        """
        Create an embedding memory set from a dataset configuration and memory configuration.
        The memory configuration requires to contain the following keys: `[key, key_tag, value, value_tag]`
        """ 
        dataset = build_dataset(dataset_config)

        key = memory_config['key']
        key_tag = memory_config['key_tag']
        value = memory_config['value']
        value_tag = memory_config['value_tag']

        subset_file = memory_config['subset'] if "subset" is memory_config else None            

        return cls(dataset, key, key_tag, subset_file=subset_file)#, value, value_tag)

    def __init__(self, dataset: DatasetBase, 
                 key: Literal['image'], 
                 key_tag: tuple[str, str, str],
                 subset_file=None,
                 value: None|Literal['image', 'depth']=None, 
                 value_tag: None|tuple[str, str, str]=None, 
                ):
        
        self.dataset = dataset

        self.key, self.key_tag = key, key_tag
        self.value, self.value_tag = value, value_tag

        self.indices = range(len(dataset))
        if subset_file is not None:
            self.indices = torch.load(subset_file)

        key_mem = [Embedding.from_dataset(self.get_path(idx, key), *key_tag).embedding for idx in tqdm(self.indices)]
        key_mem = torch.stack(key_mem, dim=0)
        self.key_mem = key_mem / key_mem.norm(dim=-1, keepdim=True)
        self.value_mem = None

        if value is not None and value_tag is not None:
            value_mem = [Embedding.from_dataset(self.get_path(idx, value), *value_tag).embedding for idx in tqdm(range(len(dataset)))] 
            value_mem = torch.stack(value_mem, dim=0)
            self.value_mem = value_mem / value_mem.norm(dim=-1, keepdim=True)

    def __len__(self):
        return len(self.indices)

    def is_rgb(self):
        return isinstance(self.dataset, DatasetBase)

    def is_rgbd(self):
        return isinstance(self.dataset, DatasetBaseDepth)

    def get_path(self, idx, modality="image"):
        if modality == "image":
            return self.dataset.get_image_path(idx)
        elif modality == "depth" and self.is_rgbd():
            return self.dataset.get_depth_path(idx)
        else:
            raise Exception(f"Path for type: {type} is not defined")

    # def search(self, embed, k, ignore_first=False, return_indices=False):

    #     # Normalize the embedding just in case
    #     norm_embed = embed / embed.norm(dim=-1, keepdim=True)
    #     sims, indices = self.index.search(norm_embed, k=k+1 if ignore_first else k)

    #     sims = sims[:,1:] if ignore_first else sims
    #     indices = indices[:,1:] if ignore_first else indices

    #     if return_indices:
    #         return self.key_mem[indices], self.value_mem[indices], indices
    #     return self.key_mem[indices], self.value_mem[indices]
        
