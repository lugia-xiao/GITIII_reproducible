#!/bin/bash



#SBATCH --job-name=BC_evaluate

#SBATCH --output=1.out

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=8

#SBATCH --gpus=a5000:1

#SBATCH --partition=gpu

#SBATCH --time=6:00:00



jupyter nbconvert --to notebook --execute evaluate.ipynb
