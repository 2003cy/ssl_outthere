#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH --nodes=2
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4        # 4 GPUs per node
#SBATCH --tasks-per-node=4       # 1 task per GPU (Lightning maps each to a rank)
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --output=/u/yacheng/ssl_outthere/jwst_dino_dist-%j.log
#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

# Multi-node x multi-GPU DDP. srun launches tasks-per-node processes per node;
# Lightning's SLURMEnvironment auto-detects SLURM_PROCID / NTASKS / NODELIST and
# wires up the process group — no custom distributed.py needed.
# Keep devices = GPUs-per-node and num_nodes = #nodes consistent with the SBATCH
# directives above. Effective batch = batch_size x devices x num_nodes; keep it at
# 768 (=96x4x2) so the pre-scaled lr in the config stays correct.

module purge
module load cuda/12.6

config="/u/yacheng/ssl_outthere/encoder_image/jwst_dino/jwst_dino.yaml"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON=/u/yacheng/ssl_outthere/.pixi/envs/h100/bin/python

cd /u/yacheng/ssl_outthere/encoder_image/jwst_dino

srun $PYTHON trainer.py fit \
    --config="$config" \
    --trainer.devices=4 \
    --trainer.num_nodes=2
