#!/bin/bash
#SBATCH --job-name=PROMOTE_Test_Training
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
# 
module load jaspy
python pipeline.py