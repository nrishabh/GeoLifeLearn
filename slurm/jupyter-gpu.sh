#!/bin/bash

#SBATCH --account=danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=jupyter-gpu
#SBATCH --output=slurm/outputs/jupyter-gpu-%J.log

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

## or use your own python/conda enviromnent
XDG_RUNTIME_DIR=””
port=$(shuf -i8000-9999 -n1)
echo $port
node=$(hostname -s)
user=$(whoami)
jupyter-notebook --no-browser --port=${port} --ip=${node}