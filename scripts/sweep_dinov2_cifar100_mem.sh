#!/bin/bash

PYTHON_SCRIPT=src/train.py
CONFIG=src/config/dinov2_cifar100_sub1000.yml

INPUT_MODALITY="image"
REPEAT_N="$1"

TEMPERATURES=(0.005)
ALPHAS=(0.5)

SUBSETS=(/project/src/data/datasets/cifar100/subset1000.pt)
# SUBSETS=(/project/src/data/datasets/cifar100/subset100.pt /project/src/data/datasets/cifar100/subset1000.pt /project/src/data/datasets/cifar100/subset2000.pt /project/src/data/datasets/cifar100/subset3000.pt /project/src/data/datasets/cifar100/subset4000.pt /project/src/data/datasets/cifar100/subset5000.pt /project/src/data/datasets/cifar100/subset6000.pt /project/src/data/datasets/cifar100/subset7000.pt /project/src/data/datasets/cifar100/subset8000.pt /project/src/data/datasets/cifar100/subset9000.pt /project/src/data/datasets/cifar100/subset10000.pt)

for TEMPERATURE in "${TEMPERATURES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for SUBSET in "${SUBSETS[@]}"; do
            echo "Running repeat $REPEAT with batch size $PYTHON_SCRIPT --temperature $TEMPERATURE --subset $SUBSET"
            python $PYTHON_SCRIPT $CONFIG \
            --model.input-modality $INPUT_MODALITY \
            --model.retrieval_augmentation.temperature $TEMPERATURE --seed 1 \
            --model.retrieval_augmentation.key image \
            --model.retrieval_augmentation.key_tag image,dinov2_large,classification \
            --model.retrieval_augmentation.value image \
            --model.retrieval_augmentation.value_tag image,dinov2_large,classification \
            --model.retrieval_augmentation.alpha $ALPHA \
            --model.retrieval_augmentation.subset $SUBSET

            if [ $? -ne 0 ]; then
                echo "Error: Python script failed with option: $OPTION"
                # Optionally, decide whether to exit or continue
                continue
            fi
        done
    done
done
