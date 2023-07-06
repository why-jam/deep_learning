#!/bin/bash
#SBATCH --partition=iirmas-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=iirmas-gpu
###SBATCH --time=2:00:00
source ~/.bashrc
conda deactivate
conda activate tensorflow
conda activate pytorch
python leaves_AlexNet.py
###python leaves_resnet50.py
