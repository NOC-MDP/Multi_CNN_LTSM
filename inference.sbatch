#!/bin/bash
#SBATCH --job-name=PROMOTE_Test_Inference
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
# 
module load jaspy
python inference.py