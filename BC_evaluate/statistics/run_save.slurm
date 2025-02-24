#!/bin/bash

#SBATCH --partition=bigmem

#SBATCH --job-name=BC_statistics

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=64G

#SBATCH --time=24:00:00

#SBATCH --output=record_value.out



jupyter nbconvert --to notebook --execute statistics.ipynb
