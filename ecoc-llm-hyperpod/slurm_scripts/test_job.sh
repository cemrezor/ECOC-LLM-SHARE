#!/bin/bash
#
# Simple Slurm job script for testing GPU usage in a new cluster
#
#SBATCH --job-name=slurm_test          # Job name
#SBATCH --output=slurm_test_%j.out     # Standard output and error log (%j expands to jobID)
#SBATCH --time=0:01:00                 # Run time limit (hh:mm:ss)
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem=100M                     # Total memory required


# Exit on error
set -e  

echo "Hello from Slurm job $SLURM_JOB_ID"
echo "Running on host: $(hostname)"

# Check GPU assignment
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
nvidia-smi

# Sleep for a few seconds to observe GPU usage
sleep 120
