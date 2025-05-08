#!/bin/bash

python src/embed.py /project/src/config/embed/imagebind_sunrgbd_classification.yml raw --num-augmentations 0 --batch-size 32 --dataset.split train
python src/embed.py /project/src/config/embed/imagebind_sunrgbd_classification.yml raw --num-augmentations 0 --batch-size 32 --dataset.split test


# Embed for segmentation on sunrgbd
# python src/embed.py src/config/embed/imagebind_sunrgbd.yml segmentation --num-augmentations 32 --batch-size 32 --dataset.split train
# python src/embed.py src/config/embed/imagebind_sunrgbd.yml segmentation --num-augmentations 32 --batch-size 32 --dataset.split test