#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 167:59:59
#SBATCH -o test_%j.out
#SBATCH -J test
#SBATCH --mail-user=cskate@wpi.edu
#SBATCH --mail-type=ALL

echo "DL Job running on $(hostname)"

echo "Loading Python Virtual Environment"

source ~/deep_learning/tempenv/bin/activate

echo "Running Python Code"

python3 scripts/main.py -d ../dataset
