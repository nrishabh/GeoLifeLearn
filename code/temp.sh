#!/bin/bash

#SBATCH --account=danielk
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 4:00:00
#SBATCH --job-name zipxtract
#SBATCH --output zipxtract_%j.log

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

python /home/rnanawa1/GeoLifeLearn/code/unzip.py