#!/usr/bin/env bash
set -ex

TARGET_DIR="/fsx/ubuntu/ecoc-llm-env"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p "$TARGET_DIR/miniconda3"

source "$TARGET_DIR/miniconda3/bin/activate"

conda create -y -p "$TARGET_DIR/python-venv" python=3.9

source activate "$TARGET_DIR/python-venv/"

conda install -y pytorch=2.5 transformers datasets pytorch-cuda=12.4 -c pytorch -c nvidia

mkdir -p "$TARGET_DIR/checkpoints"
mkdir -p "$TARGET_DIR/code"
mkdir -p "$TARGET_DIR/data"

echo "Environment setup complete in $TARGET_DIR."