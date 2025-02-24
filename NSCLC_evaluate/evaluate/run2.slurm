#!/bin/bash



#SBATCH --job-name=NSCLC_evaluate2

#SBATCH --output=2.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=25G

#SBATCH --gpus=a5000:1

#SBATCH --partition=gpu_devel

#SBATCH --time=6:00:00



jupyter nbconvert --to notebook --execute evaluate1.ipynb
