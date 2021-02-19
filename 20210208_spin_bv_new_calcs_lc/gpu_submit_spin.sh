#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=12:00:00
#SBATCH --job-name=new_spin_lc
#SBATCH --nodes=3
#SBATCH --gres=gpu:2
# TODO why use scratch for the log file?
#SBATCH --output=/scratch/jlaw/rlmolecule/20210215_spin_bv_lc/2021-02-17-gpu.%j.out
# Can't use bash variables inside of the SBATCH options
#model_name="20210216_spin_bv_lc"
#outputs_dir="outputs/$model_name/log"
#mkdir -p outputs_dir
#curr_date=`date +%F`
##SBATCH --output=$outputs_dir/$curr_date-gpu.%j.out

source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
module load cudnn/7.6.1  # ensure the correct drivers are loaded to use the GPUs
conda activate /projects/rlmolecule/jlaw/envs/tf2_gpu

echo "Started at `date`"
echo `which python`

for ((i = 0 ; i < 6 ; i++)); do
srun -l -n 1 --gres=gpu:1 --nodes=1 python train_model_spin_bv.py $i &
#srun python train_model_spin_bv.py $1
done

wait

echo "Finished at `date`"
