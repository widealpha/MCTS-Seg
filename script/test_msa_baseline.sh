#!/bin/bash
# Script to test Medical SAM Adapter

dataset=ISIC2018
image_size=512
# 定义一个数组
epochs=(1 2 3 4 5 10 15 20)
train_time=2025_06_24_00_13_51
log_time=$(date +"%Y%m%d_%H%M%S")
encoder="vit_h"

cd ../
# 循环读取数组
for epoch in "${epochs[@]}"; do
    echo "Current epoch: $epoch"
    msa_ckpt="/home/kmh/ai/MCTS-Seg/src/baseline/Medical-SAM-Adapter/logs/msa-${dataset}-${image_size}-${encoder}_${train_time}/Model/epoch${epoch}_checkpoint.pth"
    # msa_ckpt="/home/kmh/ai/MCTS-Seg/src/baseline/Medical-SAM-Adapter/logs/msa-ISIC2016-512_2025_06_21_23_38_30/Model/epoch${epoch}_checkpoint.pth"
    
    echo "Running MSA baseline with dataset: ${dataset}, image size: ${image_size}, MSA checkpoint: ${msa_ckpt}"


    log_dir="logs/test/${log_time}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/msa_baseline_test_epoch${epoch}.log"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: None, Point num: 1" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="random" \
        -point_num=1 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: random, Point num: 1" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="random" \
        -point_num=2 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: random, Point num: 2" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="random" \
        -point_num=5 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: random, Point num: 5" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="random" \
        -point_num=10 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: random, Point num: 10" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="random" \
        -point_num=20 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: random, Point num: 20" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="box" \
        -point_num=1 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: box, Point num: 1" | tee -a "$log_file"

    python src/baseline/msa_baseline.py \
        -dataset="${dataset}" \
        -image_size=${image_size} \
        -point_type="centroid" \
        -point_num=1 \
        -msa_ckpt="${msa_ckpt}" | tee -a "$log_file"
    echo "Done: Point type: centroid, Point num: 1" | tee -a "$log_file"

done

