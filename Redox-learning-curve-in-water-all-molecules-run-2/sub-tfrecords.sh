#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --job-name=tfrecords
#SBATCH --mem=700GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --output=tfrecords.o%j
#SBATCH --error=tfrecords.e%j

source ~/.bashrc
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load conda
conda activate /projects/rlmolecule/svss/envs/tf2_gpu

srun python /home/svss/projects/Project-Redox/Redox-learning-curve-in-water-all-molecules-run-2/preprocess_inputs_redox.py