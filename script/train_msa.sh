#!/bin/bash
# Script to train Medical SAM Adapter

dataset=ISIC2018
image_size=512
encoder="vit_h"
cd "$(dirname "$0")/../src/baseline/Medical-SAM-Adapter"

echo "Training MSA with dataset: ${dataset}, image size: ${image_size}"

python train.py \
    -sam_ckpt="/home/kmh/ai/MCTS-Seg/data/external/sam_vit_h_4b8939.pth" \
    -encoder="${encoder}" \
    -exp_name="msa-${dataset}-${image_size}-${encoder}" \
    -dataset=isic \
    -image_size=${image_size} \
    -data_path="/home/kmh/ai/MCTS-Seg/data/${dataset}/raw" \
    -vis=100

