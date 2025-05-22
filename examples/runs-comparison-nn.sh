#!/bin/bash
#SBATCH -J ala
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu  
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4  
#SBATCH --mem-per-cpu=15G

module load Python/3.13.2 
python3 ala-nn-asjp.py --runs=100
python3 ala-nn-combined.py --runs=100
python3 ala-nn-grambank.py --runs=100
python3 ala-nn-lexibank.py --runs=100
