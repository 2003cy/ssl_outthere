#!/bin/bash -l
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --output=/u/yacheng/nexus/ssl_outthere/encoder_image/jwst_dino/sbatch/jwst_dino_dev-%j.log
#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

# Quick dev trial on the gpudev partition (1 node x 4 A100, 10 min). Overrides the
# schedule to tiny values so train + val + teacher-export + checkpoint all fire
# inside the time limit — this tests the pipeline, not convergence.

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
    --trainer.num_nodes=1 \
    --trainer.max_epochs=4 \
    --trainer.limit_train_batches=50 \
    --trainer.check_val_every_n_epoch=2
