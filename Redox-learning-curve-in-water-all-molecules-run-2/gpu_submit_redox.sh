#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=redox_lc
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=gpu_model.%j.out

source ~/.bashrc
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load conda
conda activate /projects/rlmolecule/svss/envs/tf2_gpu

srun -l hostname

for ((i = 0 ; i < 10 ; i++)); do
    srun -l -n 1 --gres=gpu:1 --nodes=1 python train_model_redox.py $i &
done

wait