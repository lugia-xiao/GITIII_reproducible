#!/bin/bash



#SBATCH --job-name=Mouse_brain_evaluate

#SBATCH --output=1.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --gpus=a5000:1

#SBATCH --partition=gpu_devel

#SBATCH --time=6:00:00



jupyter nbconvert --to notebook --execute evaluate.ipynb
