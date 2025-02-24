#!/bin/bash

#SBATCH --partition=bigmem

#SBATCH --job-name=cellchat

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=64G

#SBATCH --time=24:00:00

#SBATCH --output=1_run_cellchat.out



R -e "rmarkdown::render('run_CellChat.Rmd')"
