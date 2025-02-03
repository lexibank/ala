#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem 500G
#SBATCH -J results_ala
#SBATCH --partition dlcegpu
#SBATCH -w dlcenode01
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 -u ala-ff-torch.py --database=lexibank
python3 -u ala-ff-torch.py --database=grambank
python3 -u ala-ff-torch.py --database=combined
python3 -u ala-ff-torch.py --database=asjp
# python3 -u ala-ff-torch.py --database=lexibank -isolates -longdistance
#python3 -u ala-ff-torch.py --database=asjp -isolates -longdistance
#python3 -u ala-ff-torch.py --database=grambank -isolates -longdistance
#python3 -u ala-ff-torch.py --database=combined -isolates -longdistance
