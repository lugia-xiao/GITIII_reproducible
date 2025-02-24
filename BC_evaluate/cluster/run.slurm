#!/bin/bash

#SBATCH --partition=week

#SBATCH --job-name=BC_plot_all_softmax

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=64G

#SBATCH --time=165:00:00

#SBATCH --output=plot.out


jupyter nbconvert --to notebook --execute cluster_cell_type.ipynb
