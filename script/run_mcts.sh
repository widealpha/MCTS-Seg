#!/bin/bash
# Script to test Medical SAM Adapter

dataset=ISIC2016
image_size=512
msa_ckpt='/home/kmh/ai/MCTS-Seg/src/baseline/Medical-SAM-Adapter/logs/msa-ISIC2016-512_2025_06_21_23_38_30/Model/epoch3_checkpoint.pth'

cd ../

echo "Running MCTS with dataset: ${dataset}, image size: ${image_size}, MSA checkpoint: ${msa_ckpt}"

python src/models/mcts.py \
    -dataset="${dataset}" \
    -image_size=${image_size} \
    -msa_ckpt="${msa_ckpt}" \
    -bone="msa"

