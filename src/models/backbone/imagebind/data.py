#!/usr/bin/env python3
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from typing import Union

# from transforms.parser import load_and_convert_depth


def load_and_transform_vision_data(image_sources: list[str | Path | Image.Image], device):
    if image_sources is None:
        return None

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_source in image_sources:
        if isinstance(image_source, (str, Path)):
            with open(image_source, "rb") as fopen:
                image = Image.open(fopen).convert("RGB")
        elif isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        else:
            image = image_source

        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)

def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=return_bpe_path())
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_depth(depth_sources: list[Union[str, Path, Image.Image]], focal_lengths, baselines, device):
    if depth_sources is None:
        return None
    
    depth_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )

    for i, depth_source in enumerate(depth_sources):
        if isinstance(depth_source, (str, Path)):
            with open(depth_source, "rb") as fopen:
                depth_image = load_and_convert_depth(depth_source)
        elif isinstance(depth_source, Image.Image):
            depth_image = depth_source.convert("I")
        else:
            depth_image = depth_source

        depth_image = convert_depth_to_disparity(depth_image, focal_lengths[i], baselines[i]).unsqueeze(0)
        depth_image = data_transform(depth_image).to(device)
        depth_outputs.append(depth_image)
    return torch.stack(depth_outputs, dim=0)

def convert_depth_to_disparity(depth_image, focal_length: float, baseline: float, min_depth=0.01, max_depth=50):
    depth = np.array(depth_image).astype(np.float32)
    if min_depth is not None:
        depth = depth.clip(min=min_depth, max=max_depth)
    disparity = baseline * focal_length / depth
    return torch.from_numpy(disparity).float()

def return_bpe_path():
    return str(Path("models/backbone/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz").resolve())
