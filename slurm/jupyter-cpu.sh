#!/bin/bash

#SBATCH --account=danielk
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-cpu
#SBATCH --output slurm/outputs/jupyter-cpu-%J.log

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

## or use your own python/conda enviromnent
XDG_RUNTIME_DIR=””
port=$(shuf -i8000-9999 -n1)
echo $port
node=$(hostname -s)
user=$(whoami)
jupyter-notebook --no-browser --port=${port} --ip=${node}