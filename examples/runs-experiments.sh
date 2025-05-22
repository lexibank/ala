#!/bin/bash
#SBATCH -J results-ala
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu  
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4  
#SBATCH --mem-per-cpu=10G

module load Python/3.13.2 
python3 ala-nn-combined.py --experiment --runs=100
python3 ala-nn-grambank.py --experiment --runs=100
python3 ala-nn-lexibank.py --experiment --runs=100
