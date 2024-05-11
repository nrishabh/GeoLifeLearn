#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=cs601-llm-prompt-recovery
#SBATCH --output=slurm/outputs/vit-base-patch16-224-%J.log
#SBATCH --mail-type=all 
#SBATCH --mail-user=rnanawa1@jhu.edu

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

python /home/rnanawa1/GeoLifeLearn/experiment.py "google/vit-base-patch16-224" --learning_rate 0.00002 --epochs 10 --data_dir "/home/rnanawa1/GeoLifeLearn/data/species25/" --batch_size 32