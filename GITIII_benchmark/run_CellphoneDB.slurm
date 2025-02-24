#!/bin/bash

#SBATCH --partition=bigmem

#SBATCH --job-name=CellphoneDB

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=64G

#SBATCH --time=24:00:00

#SBATCH --output=1_run_CellphoneDB.out



jupyter nbconvert --to notebook --execute run_CellphoneDB.ipynb
