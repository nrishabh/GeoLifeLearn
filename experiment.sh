#!/bin/bash

# charges to Daniel Khashabi's GPU account
#SBATCH --account=danielk_gpu

# you can choose between a100, v100 and ica100
#SBATCH --partition=a100

# no. of gpus you need
#SBATCH --gres=gpu:1

# no. of nodes you need
#SBATCH --nodes=1

# no. of tasks you want to allow per node
#SBATCH --ntasks-per-node=1

# memory needed
#SBATCH --mem=16G

# max. time for the job. it will timeout after this
#SBATCH --time=4:00:00

# your job name
#SBATCH --job-name=cs601-llm-prompt-recovery

# the file where your output will be printed
# writing %J will replace it with the job_id during runtime
#SBATCH --output=slurm/outputs/%J.log

# if you want to be notified when your job starts and ends, use this command
#SBATCH --mail-type=all 

# if you are using the email notification, this is your email (it needs to be @jhu.edu)
#SBATCH --mail-user=rnanawa1@jhu.edu

ml anaconda
conda activate /home/rnanawa1/.conda/envs/geolife

python /home/rnanawa1/GeoLifeLearn/experiment.py "google/vit-base-patch32-224-in21k" --learning_rate 0.00002 --epochs 1 --data_dir "/home/rnanawa1/GeoLifeLearn/data/species25/" --batch_size 32