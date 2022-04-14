#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64g
#SBATCH --gres=gpu:2
#SBATCH -C K20
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -o dataparse_%j.out
#SBATCH -J dataparse

echo "Test Job running on $(hostname):"

echo "Loading Virtual Environment..."

. /opt/ohpc/pub/apps/anaconda/anaconda3/etc/profile.d/conda.sh

conda activate /work/barane/envs/semseg_env/

echo "Virtual Environment Load Successful"

# python3 /work/barane/semantic_segmentation/scripts/test/pcd_to_spherical.py -d /work/barane/data_odometry_velodyne/dataset/

python3 /work/barane/semantic_segmentation/scripts/test/load_dataset.py

