#!/bin/bash
#SBATCH --job-name="PROMOTE"
#SBATCH --time=24:00:00
#SBATCH --mem=128Gb
#SBATCH --account=nemo
#SBATCH --partition=standard
#SBATCH --qos=standard
# 
module load jaspy
python get_data.py