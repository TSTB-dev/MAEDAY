#!/bin/bash

OUTPUT_DIR="./weights"
mkdir -p "$OUTPUT_DIR"

echo "Downloading weights..."
echo "[1/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth -P "$OUTPUT_DIR"
echo "[2/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth -P "$OUTPUT_DIR"
echo "[3/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_huge.pth -P "$OUTPUT_DIR"

echo "[4/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth -P "$OUTPUT_DIR"

echo "[5/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth -P "$OUTPUT_DIR"
echo "[6/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth -P "$OUTPUT_DIR"
echo "[7/7] Downloading weights..."
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge_full.pth -P "$OUTPUT_DIR"

echo "Finished downloading weights."
