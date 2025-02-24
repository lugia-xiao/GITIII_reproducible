#!/bin/bash

#SBATCH --partition=week

#SBATCH --job-name=BC_plot1_aggregate

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=64G

#SBATCH --time=124:00:00

#SBATCH --output=aggregate.out

jupyter nbconvert --to notebook --execute cluster_aggregated_cell.ipynb
