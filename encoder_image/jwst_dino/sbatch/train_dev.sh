#!/bin/bash -l
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --output=jwst_dino_dev-%j.log
#SBATCH --mail-type=none


# Dev trial on the gpudev partition (1 node x 4 A100, 10 min walltime). 

module purge
module load cuda/12.6

# Paths are derived from this script's own location, so the launcher works
# from any checkout.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

config="$REPO/encoder_image/jwst_dino/jwst_dino.yaml"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHON="$REPO/.pixi/envs/default/bin/python"
CONDA_PREFIX="$REPO/.pixi/envs/default"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$REPO/encoder_image/jwst_dino"

srun $PYTHON trainer.py fit \
    --config="$config" \
    --trainer.devices=4 \
    --trainer.num_nodes=1 \    
    --trainer.max_epochs=50 \
    --trainer.limit_train_batches=10 \
    --trainer.check_val_every_n_epoch=1 \
    --trainer.logger.init_args.name=jwst_dino_ps6_st3_dev
