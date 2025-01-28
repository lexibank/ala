#!/bin/bash
#SBATCH --cpus-per-task 12
#SBATCH -J results_ala
#SBATCH --partition dlcegpu
#SBATCH -w dlcenode01
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 -u ala-ff-torch.py --input=asjp -intersection

#python3 -u ala-ff-torch.py --input=grambank
#python3 -u ala-ff-torch.py --input=combined
#python3 -u ala-ff-torch.py --input=asjp
# python3 -u ala-ff-torch.py --input=lexibank -isolates -longdistance
#python3 -u ala-ff-torch.py --input=asjp -isolates -longdistance
#python3 -u ala-ff-torch.py --input=grambank -isolates -longdistance
#python3 -u ala-ff-torch.py --input=combined -isolates -longdistance
