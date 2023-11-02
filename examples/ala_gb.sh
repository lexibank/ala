#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH -J results_ala
#SBATCH --partition dlcegpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 ala-grambank-ff-torch.py
