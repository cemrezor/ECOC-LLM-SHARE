#!/bin/bash
#SBATCH --job-name=gpt2_eval
#SBATCH --output=gpt2_eval.out
#SBATCH --error=gpt2_eval.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com  # Change this to your email

# Load necessary modules
module load python/3.8
module load cuda/11.3  # Adjust based on your cluster

# Activate virtual environment if needed
source /fsx/ubuntu/ecoc-llm-env/bin/activate

# Define parameters
MODEL_NAME="gpt-15M"
CHECKPOINT_PATH="/fsx/ubuntu/ecoc-llm-env/checkpoints/sha-training-gpt-15M-vocab-3000-epochs-1-epoch-1.bin"
DATASET_NAME="/fsx/ubuntu/ecoc-llm-env/data/test"
TOKENIZER_PATH="tokens.json"
OUTPUT_CSV="top_k_accuracy.csv"

# Run the Python script
python gpt2_eval.py \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_name "$DATASET_NAME" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_csv "$OUTPUT_CSV"