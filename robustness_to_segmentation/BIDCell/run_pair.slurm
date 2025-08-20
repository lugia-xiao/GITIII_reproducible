#!/bin/bash

#SBATCH --partition=bigmem

#SBATCH --job-name=CCI_pair

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=64G

#SBATCH --time=24:00:00

#SBATCH --output=1_CCI_pair_visualize.out



jupyter nbconvert --to notebook --execute downstream.ipynb
