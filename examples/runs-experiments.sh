#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem 500G
#SBATCH -J results-ala
#SBATCH --partition dlcegpu
#SBATCH -w dlcenode01
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python -u ala-nn-combined.py --experiment --runs=100
python -u ala-nn-grambank.py --experiment --runs=100
python -u ala-nn-lexibank.py --experiment --runs=100
