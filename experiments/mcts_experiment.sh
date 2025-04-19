#!/bin/bash

# Check GPU free memory
gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{if ($1 > 10240) {print $1; exit}}')
while [ -z "$gpu_memory" ]; do
    echo "Waiting for a GPU with free memory greater than 10GB..."
    sleep 5
    gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{if ($1 > 10240) {print $1; exit}}')
done
if [ -n "$gpu_memory" ]; then
    # Execute Python script if GPU free memory is greater than 10GB
    python src/models/mcts.py --dataset=ISIC2018
else
    echo "No GPU with free memory greater than 10GB available."
fi