#!/bin/bash

PYTHON_SCRIPT=src/train.py
CONFIG=src/config/dinov2_cifar100.yml

INPUT_MODALITY="image"
# REPEAT_N="$1"

TEMPERATURES=(0.00001)
REPEATS=(1)

ALPHAS=(0.99)
# ALPHAS=(0.25 0.5 0.75 1.0)
# ALPHAS=(1.0)

for REPEAT in "${REPEATS[@]}"; do

    # python $PYTHON_SCRIPT $CONFIG \
    #     --model.input-modality $INPUT_MODALITY \
    #     --seed $REPEAT \
    #     --model.retrieval_augmentation.key image \
    #     --model.retrieval_augmentation.key_tag image,dinov2_large,classification \
    #     --model.retrieval_augmentation.value image \
    #     --model.retrieval_augmentation.value_tag image,dinov2_large,classification \
    #     --model.retrieval_augmentation.alpha 0.0

    # if [ $? -ne 0 ]; then
    #     echo "Error: Python script failed with option: $OPTION"
    #     # Optionally, decide whether to exit or continue
    #     continue
    # fi

    for TEMPERATURE in "${TEMPERATURES[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            echo "Running repeat $REPEAT with batch size $PYTHON_SCRIPT --temperature $TEMPERATURE"
            python $PYTHON_SCRIPT $CONFIG \
            --model.input-modality $INPUT_MODALITY \
            --model.retrieval_augmentation.temperature $TEMPERATURE --seed $REPEAT \
            --model.retrieval_augmentation.key image \
            --model.retrieval_augmentation.key_tag image,dinov2_large,classification \
            --model.retrieval_augmentation.value image \
            --model.retrieval_augmentation.value_tag image,dinov2_large,classification \
            --model.retrieval_augmentation.alpha $ALPHA

            if [ $? -ne 0 ]; then
                echo "Error: Python script failed with option: $OPTION"
                # Optionally, decide whether to exit or continue
                continue
            fi
        done
    done
done