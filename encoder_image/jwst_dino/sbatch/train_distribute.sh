#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH --nodes=5
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4        # 4 GPUs per node
#SBATCH --tasks-per-node=4       # 1 task per GPU (Lightning maps each to a rank)
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --output=/u/yacheng/nexus/ssl_outthere/encoder_image/jwst_dino/sbatch/jwst_dino_dist-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yacheng@mpia.de

# Multi-node x multi-GPU DDP. srun launches tasks-per-node processes per node;
# Lightning's SLURMEnvironment auto-detects SLURM_PROCID / NTASKS / NODELIST and
# wires up the process group — no custom distributed.py needed.
# Keep devices = GPUs-per-node and num_nodes = #nodes consistent with the SBATCH
# directives above. Effective batch = 64 x 4 x 5 = 1280.
# LR is auto-scaled in the model: _effective_lr = lr * sqrt(batch*world_size/1024)
#   = lr * sqrt(64*20/1024) = lr * 1.118  (world_size = 4 GPU x 5 nodes = 20)

module purge
module load cuda/12.6

config="/u/yacheng/ssl_outthere/encoder_image/jwst_dino/jwst_dino.yaml"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON=/u/yacheng/ssl_outthere/.pixi/envs/default/bin/python
CONDA_PREFIX=/u/yacheng/ssl_outthere/.pixi/envs/default
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /u/yacheng/ssl_outthere/encoder_image/jwst_dino

srun $PYTHON trainer.py fit \
    --config="$config" \
    --trainer.devices=4 \
    --trainer.num_nodes=5
