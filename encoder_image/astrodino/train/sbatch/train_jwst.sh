#!/bin/bash -l
#SBATCH -t 13:00:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --tasks-per-node=4  #   running 4 tasks per node, each with 1 GPU.
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=500000
#SBATCH --output=/u/yacheng/ssl_outthere/train_test-%j.log

#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load anaconda/3/2023.03
module load cuda/12.6
#module load gcc

run_name="astrodino_f150w_vitl"
config="/u/yacheng/ssl_outthere/encoder_image/astrodino/train/configs/astrodino_jwst_vitl_f150w.yaml"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

conda activate astrodino

cd /u/yacheng/ssl_outthere/encoder_image/astrodino/train

srun python -m trainer \
    --config-file=$config --run-name=$run_name