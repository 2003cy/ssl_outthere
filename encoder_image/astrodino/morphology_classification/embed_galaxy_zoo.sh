#!/bin/bash

# define config_file list (three different configurations)
configs=(
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_7e5_vit_base/config.yaml'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_7e5_vit_large/config.yaml'
)

# define pretrained_weights list (three different weights, corresponding to the above configurations)
weights=(
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_7e5_vit_base/eval/training_59999/teacher_checkpoint.pth'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_7e5_vit_large/eval/training_59999/teacher_checkpoint.pth'
)

# submit a single job to compute embeddings for all models and aggregate into one HDF5
echo "Submitting job to compute and aggregate embeddings for ${#configs[@]} models"

sbatch <<EOF
#!/bin/bash -l
#SBATCH -t 03:45:00
#SBATCH --nodes=1

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH --output=/ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/morphology_classification/embed_galaxy_zoo-%j.log
#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load anaconda/3/2023.03
module load cuda/12.6

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

source ~/.bashrc

conda activate astrodino

export PATH=$CONDA_PREFIX/bin:$PATH

# Debug: Check which conda and python
echo "Conda default env: $CONDA_DEFAULT_ENV"
which python
python --version

srun python /ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/morphology_classification/embed_galaxy_zoo.py --model_cfgs ${configs[@]} --model_weights ${weights[@]} --max_samples 100000
EOF