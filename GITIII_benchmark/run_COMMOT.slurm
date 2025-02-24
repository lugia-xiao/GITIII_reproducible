#!/bin/bash

#SBATCH --partition=bigmem

#SBATCH --job-name=COMMOT

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=64G

#SBATCH --time=24:00:00

#SBATCH --output=1_run_COMMOT.out



jupyter nbconvert --to notebook --execute run_COMMOT.ipynb
