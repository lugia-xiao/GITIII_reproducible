#!/bin/bash

#SBATCH --partition=week

#SBATCH --job-name=NSCLC_plot1_softmax

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=64G

#SBATCH --time=124:00:00

#SBATCH --output=plot.out

jupyter nbconvert --to notebook --execute cluster_type.ipynb
jupyter nbconvert --to notebook --execute cluster_group.ipynb
