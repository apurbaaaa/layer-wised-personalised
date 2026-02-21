#!/bin/bash
# =============================================================
#  RunPod Setup Script — ISIC 2019 SwinV2 Training
#
#  Run this on a fresh RunPod instance (A40 / A100 recommended):
#    bash setup_runpod.sh
#
#  After setup completes, training starts automatically.
# =============================================================

set -euo pipefail

echo "=============================================="
echo "  ISIC 2019 — RunPod Setup"
echo "=============================================="

# ---- System deps ------------------------------------------------
apt-get update -qq && apt-get install -y -qq wget unzip > /dev/null 2>&1
echo "[1/6] System deps installed."

# ---- Python env --------------------------------------------------
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt > /dev/null 2>&1
echo "[2/6] Python packages installed."

# ---- Download dataset --------------------------------------------
DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "[3/6] Downloading ISIC 2019 dataset ..."

# CSVs (small)
wget -q -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
wget -q -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv
wget -q -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
wget -q -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv
echo "  CSVs downloaded."

# Training images (~9 GB)
if [ ! -d "$DATA_DIR/ISIC_2019_Training_Input" ]; then
    wget -q --show-progress -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
    echo "  Unzipping training images ..."
    unzip -q "$DATA_DIR/ISIC_2019_Training_Input.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/ISIC_2019_Training_Input.zip"
else
    echo "  Training images already exist, skipping."
fi

# Test images (~3 GB)
if [ ! -d "$DATA_DIR/ISIC_2019_Test_Input" ]; then
    wget -q --show-progress -P "$DATA_DIR" https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip
    echo "  Unzipping test images ..."
    unzip -q "$DATA_DIR/ISIC_2019_Test_Input.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/ISIC_2019_Test_Input.zip"
else
    echo "  Test images already exist, skipping."
fi

echo "[4/6] Dataset ready."

# ---- Verify GPU --------------------------------------------------
echo "[5/6] GPU check:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)')
else:
    print('  WARNING: No CUDA GPU detected!')
"

# ---- Start training ----------------------------------------------
echo "[6/6] Starting training ..."
echo ""

python train.py \
    --device cuda:0 \
    --batch_size 48 \
    --accum_steps 1 \
    --workers 12 \
    --epochs 25 \
    --lr 4e-4 \
    --compile \
    --mixup

echo ""
echo "Training complete! Run evaluate.py for test set results."
