#!/bin/bash
#SBATCH --job-name=python_job_array
#SBATCH --output=res_folder/simu1-1/output_%A%a.log
#SBATCH --error=res_folder/simu1-1/error_%A%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=365-00:00:00
#SBATCH --mem=30G

python -u simu1/simu1-1.py $SLURM_ARRAY_TASK_ID