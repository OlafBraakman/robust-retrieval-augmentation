#!/bin/bash

PYTHON_SCRIPT=src/train.py
CONFIG=src/config/imagebind.yml

INPUT_MODALITY="image"
REPEAT_N="$1"

TEMPERATURES=(0.0001)
REPEATS=(1 2 3)

ALPHAS=(1.0)

for REPEAT in "${REPEATS[@]}"; do
    for TEMPERATURE in "${TEMPERATURES[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            echo "Running repeat $REPEAT with batch size $PYTHON_SCRIPT --temperature $K"
            python $PYTHON_SCRIPT $CONFIG \
            --model.input-modality $INPUT_MODALITY \
            --model.retrieval_augmentation.temperature $TEMPERATURE --seed $REPEAT \
            --model.retrieval_augmentation.key image \
            --model.retrieval_augmentation.key_tag image,imagebind_huge,classification \
            --model.retrieval_augmentation.value depth \
            --model.retrieval_augmentation.value_tag depth,imagebind_huge,classification \
            --model.retrieval_augmentation.alpha $ALPHA

            if [ $? -ne 0 ]; then
                echo "Error: Python script failed with option: $OPTION"
                # Optionally, decide whether to exit or continue
                continue
            fi

            python $PYTHON_SCRIPT $CONFIG \
            --model.input-modality $INPUT_MODALITY \
            --model.retrieval_augmentation.temperature $TEMPERATURE --seed $REPEAT \
            --model.retrieval_augmentation.key image \
            --model.retrieval_augmentation.key_tag image,imagebind_huge,classification \
            --model.retrieval_augmentation.value image \
            --model.retrieval_augmentation.value_tag image,imagebind_huge,classification \
            --model.retrieval_augmentation.alpha $ALPHA

            if [ $? -ne 0 ]; then
                echo "Error: Python script failed with option: $OPTION"
                # Optionally, decide whether to exit or continue
                continue
            fi
        done
    done
done