# Retrieval augmentation for scale-sensitive scene classification tasks

This is the official repository for "Retrieval augmentation for scale-sensitive scene classification tasks"

# 1. Installation
You can use Docker (1.1) or install manually (1.2).

Use of a GPU is highly recommended

## 1.1 Docker
See `docker/Dockerfile` for the image

## 1.2 Manual install
This repository was built with **Python 3.11** and PyTorch Fabric, not sure if it works on older versions

Run `pip install -r requirements.txt`

## 2.3 Datasets
This repository mainly works with the SUNRGB-D dataset

Install here: TODO

Extra analysis is done with InSpaceType

Install here: TODO

# 2. Reproducability

This repository has two main components: 1) pre-computing image/depth embedding and 2) training/evaluating models

## 2.1 Feature embedding

`python src/embed.py --dataset [dataset] --model [model] --num-augmentations [num] --batch-size [num]`
