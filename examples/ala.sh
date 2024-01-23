#!/bin/bash
#SBATCH --cpus-per-task 12
#SBATCH -J results_ala
#SBATCH --partition dlcegpu
#SBATCH -w dlcenode04
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 -u ala-ff-torch.py --data=combined

