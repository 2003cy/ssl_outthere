#!/bin/bash -l
#SBATCH -t 00:06:00
#SBATCH --nodes=1
#SBATCH --partition=gpudev  # Use the GPU development partition
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --tasks-per-node=4  #   running 4 tasks per node, each with 1 GPU.
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=500000
#SBATCH --output=/ptmp/yacheng/outthere_ssl/eval-%j.log

#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load anaconda/3/2023.03
module load cuda/12.6
#module load gcc

run_name="astrodino_3z_band_stable_multi_epochs"
config="/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable_multi_epochs/config.yaml"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

conda activate astrodino

cd /ptmp/yacheng/outthere_ssl/AstroCLIP

srun python -m astroclip.astrodino.trainer \
    --config-file=$config --run-name=$run_name --eval-only True