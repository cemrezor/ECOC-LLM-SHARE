#!/bin/bash
#
# Slurm job script for running the training script
#
#SBATCH --job-name=gpt-training         # Job name
#SBATCH --output=/fsx/ubuntu/ecoc-llm-env/logs/train_%j.log           # Standard output and error log (%j expands to jobID)
#SBATCH --error=/fsx/ubuntu/ecoc-llm-env/logs/train_%j.log            # Standard error log
#SBATCH --time=4:00:00                  # Run time limit (hh:mm:ss)
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

cd /fsx/ubuntu/ecoc-llm-env/code

#pip install -r requirements.txt

# Run the training script
python start_training.py --model-config gpt-30M

echo "Job $SLURM_JOB_ID completed."

