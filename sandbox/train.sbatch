#!/bin/bash
#SBATCH --job-name=PROMOTE_Test_Run
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
# 
module load jaspy
python Multi_CNN_LTSM.py