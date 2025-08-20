#!/bin/bash



#SBATCH --job-name=BIDcell

#SBATCH --output=1_estimator.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=25G

#SBATCH --gpus=a5000:1

#SBATCH --partition=gpu

#SBATCH --time=48:00:00



jupyter nbconvert --to notebook --execute run.ipynb
