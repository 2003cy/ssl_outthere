#!/bin/bash -l
#SBATCH -t 21:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --tasks-per-node=4  #   running 4 tasks per node, each with 1 GPU.
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=500000
#SBATCH --output=/u/yacheng/projects/ssl_outthere/train_test-%j.log

#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load cuda/12.6

run_name="astrodino_f150w_vitb_ps6_st3_bs128"
group_name="single"
config="/u/yacheng/ssl_outthere/encoder_image/astrodino/train/configs/astrodino_f150w_vitb_ps6_st3_bs128.yaml"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON=/u/yacheng/ssl_outthere/.pixi/envs/default/bin/python

cd /u/yacheng/ssl_outthere/encoder_image/astrodino/train

srun $PYTHON -m trainer \
    --config-file="$config" \
    --run-name="$run_name" \
    --group-name="$group_name"
