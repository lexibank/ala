#!/bin/bash -l

#SBATCH -J ALA
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#SBATCH --time=24:00:00

module load intel/21.2.0 impi/2021.2 cuda/11.2 anaconda/3/2023.03

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 -u ala-ff-torch.py --data=lexibank
#python3 -u ala-ff-torch.py --data=grambank
#python3 -u ala-ff-torch.py --data=combined
#python3 -u ala-ff-torch.py --data=asjp
# python3 -u ala-ff-torch.py --data=lexibank -isolates -longdistance
#python3 -u ala-ff-torch.py --data=asjp -isolates -longdistance
#python3 -u ala-ff-torch.py --data=grambank -isolates -longdistance
#python3 -u ala-ff-torch.py --data=combined -isolates -longdistance
