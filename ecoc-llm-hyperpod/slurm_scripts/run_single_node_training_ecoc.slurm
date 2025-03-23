#!/bin/bash
#
# Slurm job script for running the training script
#
#SBATCH --job-name=gpt-training         # Job name
#SBATCH --output=/fsx/ubuntu/ecoc-llm-env/logs/train_ecoc_%j.log           # Standard output and error log (%j expands to jobID)
#SBATCH --error=/fsx/ubuntu/ecoc-llm-env/logs/train_ecoc_%j.log            # Standard error log
#SBATCH --time=24:00:00                  # Run time limit (hh:mm:ss)
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Total memory required

echo "Starting Slurm job $SLURM_JOB_ID"
echo "Running on host: $(hostname)"

# Load any necessary modules
#module load python/3.9                      # Adjust to the Python module on your cluster

# Activate the virtual/conda environment
source /fsx/ubuntu/ecoc-llm-env/miniconda3/bin/activate
conda activate /fsx/ubuntu/ecoc-llm-env/python-venv

cd /fsx/ubuntu/wyuhuang/shamika/ECOC-LLM/ecoc-llm-hyperpod/code/

#pip install -r requirements.txt

# Run the training script
python start_training_ecoc.py --model-config gpt-15M --ecoc-type minimal

echo "Job $SLURM_JOB_ID completed."

