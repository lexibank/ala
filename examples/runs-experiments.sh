#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem 500G
#SBATCH -J results_ala
#SBATCH --partition dlcegpu
#SBATCH -w dlcenode01
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python -u ala_nn_asjp.py --intersection=False --runs=100
python -u ala_nn_combined.py --intersection=False --runs=100
python -u ala_nn_grambank.py --intersection=False --runs=100
python -u ala_nn_lexibank.py --intersection=False --runs=100
