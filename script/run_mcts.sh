#!/bin/bash
# Script to test Medical SAM Adapter

dataset=ISIC2016
image_size=512
train_time=2025_06_23_22_23_11
encoder="vit_h"
epoch=10
msa_ckpt="/home/kmh/ai/MCTS-Seg/src/baseline/Medical-SAM-Adapter/logs/msa-${dataset}-${image_size}-${encoder}_${train_time}/Model/epoch${epoch}_checkpoint.pth"
cd "$(dirname "$0")/../"

echo "Running MCTS with dataset: ${dataset}, image size: ${image_size}, MSA checkpoint: ${msa_ckpt}"

python src/models/mcts.py \
    -dataset="${dataset}" \
    -image_size=${image_size} \
    -bone="sam" \
    -msa_ckpt="${msa_ckpt}" \
    -bone="msa"
