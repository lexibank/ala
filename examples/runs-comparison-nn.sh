#!/bin/bash
#SBATCH -J results-ala
#SBATCH --partition=gpu  
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4  
#SBATCH --mem-per-cpu=15G

module load Python/3.13.2 
python ala-nn-asjp.py --runs=100
python ala-nn-combined.py --runs=100
python ala-nn-grambank.py --runs=100
python ala-nn-lexibank.py --runs=100
