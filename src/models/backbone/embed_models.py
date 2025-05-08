from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import cast

import torch.hub as hub
import abc
from .imagebind.models.imagebind_model import imagebind_huge
from typing import Literal
import torchvision.models

from torchvision.transforms import v2

from types import SimpleNamespace

EmbedModelType = SimpleNamespace(
    DINOV2_B="dinov2_big",
    DINOV2_L="dinov2_large",
    DINOV2_G="dinov2_giant",
    IMAGEBIND_HUGE="imagebind_huge",
    RESNET_50="resnet_50",
)

class BackboneModel(abc.ABC, pl.LightningModule):
    
    @staticmethod
    def fromname(type, device="cuda"):
        if type.split("_")[0] == "dinov2":
            return DinoV2Backbone(type, device=device)
        elif type.split("_")[0] == "imagebind":
            return ImageBindBackbone(type, device=device)
        elif type.split("_")[0] == "resnet":
            return ResNetBackbone(type, device=device)
        else:
            raise Exception(f"Model with type: {type} not found")

    def __init__(self, input_modalities=["image"]) -> None:
        super().__init__()
        self.input_modalities = input_modalities
        self.no_grad = False

    @abc.abstractmethod
    def preprocess(self, sample, modality="image") -> Any:
        """Responsible for shifting fully transformed data to a format understandable by the model, 
        such as adding to a dict for a different modality"""
        pass

    @abc.abstractmethod
    def postprocess(self, sample, modality="image"):
        """Return to default format"""
        pass

    @property
    @abc.abstractmethod
    def model(self) -> nn.Module:
        pass

    def forward(self, x, modality="image"):
        x = self.preprocess(x, modality=modality)

        if self.no_grad:
            with torch.no_grad():
                output = self.model(x)
        else:
            output = self.model(x)
        x = self.postprocess(output, modality=modality)
        return x / x.norm(dim=-1, keepdim=True)

class ResNetBackbone(BackboneModel):

    def __init__(self, model_type, device="cuda") -> None:
        super().__init__(input_modalities=["image"])
        if model_type == EmbedModelType.RESNET_50:
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            raise Exception(f"Model {model_type} is not supported")
        
        # Drop the final classification layer
        self._model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self._model.add_module("pool", nn.AdaptiveAvgPool2d(1))

        self._model.eval()

    @property
    def model(self):
        return self._model
    
    def preprocess(self, sample, modality="image"):
        if modality == "image":
            return sample
        if modality == "depth":
            return sample.repeat(1,3,1,1)
    
    def postprocess(self, sample, modality="image"):
        return sample.squeeze(dim=[2,3])
        # sample = sample.view(sample.size(0), -1)

        # # Not very elegant, but if only image
        # if len(self.input_modalities) > 1:
        #     image_batch, depth_batch = sample.split(sample.shape[0]//2, dim=0)
        #     return {
        #         'image': image_batch,
        #         'depth': depth_batch
        #     }
        # return {
        #     'image': sample
        # }
    
class DinoV2Backbone(BackboneModel):

    def __init__(self, model_type, device="cuda") -> None:
        super().__init__(input_modalities=["image"])
        if model_type == EmbedModelType.DINOV2_B:
            self._model = cast(nn.Module, hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')).to(device)
        elif model_type == EmbedModelType.DINOV2_L:
            self._model = cast(nn.Module, hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')).to(device)
        elif model_type == EmbedModelType.DINOV2_G:
            self._model = cast(nn.Module, hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')).to(device)
        else:
            raise Exception(f"Model {model_type} is not supported")
        self._model.eval()

    @property
    def model(self):
        return self._model
    
    def preprocess(self, sample, modality="image"):
        if modality == "image":
            return sample
        if modality == "depth":
            return sample.repeat(1,3,1,1)
    
    def postprocess(self, sample, modality="image"):
        return sample
    

class ImageBindBackbone(BackboneModel):

    def __init__(self, model_type, input_modalities=["image", "depth"], device="cuda") -> None:
        super().__init__(input_modalities=input_modalities)
        if model_type == EmbedModelType.IMAGEBIND_HUGE:
            self._model = imagebind_huge(pretrained=True).to(device)
        else:
            raise Exception(f"Model {model_type} is not supported")
        self._model.eval()

    @property
    def model(self):
        return self._model
    
    def preprocess(self, sample, modality="image"):
        value = {}
        if modality == "image":
            value['vision'] = sample
        elif modality == "depth":
            value['depth'] = sample

        return value

    def postprocess(self, sample, modality="image"):
        if modality == "image":
            return sample['vision']
        if modality == "depth":
            return sample['depth']
