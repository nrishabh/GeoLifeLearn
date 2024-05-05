#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=cs601-llm-prompt-recovery
#SBATCH --output=slurm/outputs/llm-prompt-recovery-gpu-%J.log

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

## or use your own python/conda enviromnent
python /home/rnanawa1/GeoLifeLearn/code/exp1.py
