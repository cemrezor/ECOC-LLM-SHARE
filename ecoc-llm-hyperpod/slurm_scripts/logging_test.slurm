#!/bin/bash

#SBATCH --job-name=log-test
#SBATCH --output=log_test_%j.out
#SBATCH --error=log_test_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

python python_test.py