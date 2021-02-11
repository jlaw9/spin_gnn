#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=12:00:00
#SBATCH --job-name=new_spin_lc
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/jlaw/rlmolecule/20210208_spin_bv_lc/2021-02-10-gpu.%j.out

source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/rlmolecule/pstjohn/envs/tf2_gpu

echo "Started at `date`"
echo `which python`

for ((i = 2 ; i < 6 ; i++)); do
srun -l -n 1 --gres=gpu:1 --nodes=1 python train_model_spin_bv.py $i &
#srun python train_model_spin_bv.py $1
done

wait

echo "Finished at `date`"
