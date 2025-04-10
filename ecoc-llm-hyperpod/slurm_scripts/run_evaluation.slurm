#!/bin/bash
#SBATCH --job-name=gpt2_eval
#SBATCH --output=/fsx/ubuntu/ecoc-llm-env/logs/gpt2_eval_%j.log
#SBATCH --error=/fsx/ubuntu/ecoc-llm-env/logs/gpt2_eval_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00


# Activate virtual environment if needed
source /fsx/ubuntu/ecoc-llm-env/miniconda3/bin/activate
conda activate /fsx/ubuntu/ecoc-llm-env/python-venv

cd /fsx/ubuntu/ecoc-llm-env/code

# Define parameters
MODEL_NAME="gpt-15M"
CHECKPOINT_PATH="/fsx/ubuntu/ecoc-llm-env/checkpoints/sha-training-gpt-15M-vocab-3000-epochs-1-epoch-1.bin"
DATASET_NAME="/fsx/ubuntu/ecoc-llm-env/data/validation"
TOKENIZER_PATH="tokens.json"
OUTPUT_CSV="top_k_accuracy.csv"


# Run the Python script
python eval.py \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_name "$DATASET_NAME" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_csv "$OUTPUT_CSV"