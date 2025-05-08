from tools.registry import Registry
from data.datasets.sunrgbd.pytorch_dataset import SUNRGBD, SUNRGB
from data.datasets.nyuv2.pytorch_dataset import NYUv2
from data.datasets.mitindoor.pytorch_dataset import MITIndoor
from data.datasets.eurosat.pytorch_dataset import EuroSAT
from data.datasets.gtsrb.pytorch_dataset import GTSRB
from data.datasets.cifar100.pytorch_dataset import CIFAR100
from data.datasets.imagenet.pytorch_dataset import ImageNet

from data.datasets.dataset_base import DatasetBase

# Instantiate Registry class
dataset_registry=Registry()

@dataset_registry.register('imagenet_classification')
def build_imagenet_classifcation(dataset_cfg):
    return ImageNet(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('cifar100_classification')
def build_cifar100_classifcation(dataset_cfg):
    return CIFAR100(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('gtsrb_classification')
def build_gtsrb_classifcation(dataset_cfg):
    return GTSRB(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('eurosat_classification')
def build_eurosat_classifcation(dataset_cfg):
    return EuroSAT(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('sunrgbd_classification')
def build_sunrgbd_classifcation(dataset_cfg):
    return SUNRGBD(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'], modality=dataset_cfg['modality'])

@dataset_registry.register('sunrgb_classification')
def build_sunrgb_classifcation(dataset_cfg):
    return SUNRGB(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('nyuv2_classification')
def build_nyuv2_classification(dataset_cfg):
    return NYUv2(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

@dataset_registry.register('mitindoor_classification')
def build_mitindoor_classification(dataset_cfg):
    return MITIndoor(data_dir=dataset_cfg['dir'], split=dataset_cfg['split'])

def build_dataset(dataset_cfg) -> DatasetBase:
    dataset = dataset_registry[dataset_cfg['name']](dataset_cfg)
    return dataset
