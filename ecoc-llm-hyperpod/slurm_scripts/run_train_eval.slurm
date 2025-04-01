#!/bin/bash
#SBATCH --job-name=train_and_eval
#SBATCH --output=/fsx/ubuntu/ecoc-llm-env/logs/train_and_eval_%j.log
#SBATCH --error=/fsx/ubuntu/ecoc-llm-env/logs/train_and_eval_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Starting Slurm job $SLURM_JOB_ID on host $(hostname)"

source /fsx/ubuntu/ecoc-llm-env/miniconda3/bin/activate
conda activate /fsx/ubuntu/ecoc-llm-env/python-venv

cd /fsx/ubuntu/ecoc-llm-env/code

python train_and_eval.py

echo "Job $SLURM_JOB_ID completed."
