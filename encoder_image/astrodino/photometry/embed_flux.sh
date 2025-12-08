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

# Dataset definition and output file (single HDF5 aggregating embeddings + flux for all models)
dataset_str="LegacySurvey:split=test:root=/ptmp/yacheng/outthere_ssl/:extra=\"\""
output_path='/ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/legacy_flux_embeddings.h5'

if [[ ${#configs[@]} -ne ${#weights[@]} ]]; then
    echo "configs and weights length mismatch" >&2
    exit 1
fi

echo "Submitting flux embedding job for ${#configs[@]} models"

sbatch <<EOF
#!/bin/bash -l
#SBATCH -t 02:00:00
#SBATCH --nodes=1

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100000
#SBATCH --output=/ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/embed_flux-%j.log
#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load anaconda/3/2023.03
module load cuda/12.6

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh
source ~/.bashrc
conda activate astrodino

export PATH=\$CONDA_PREFIX/bin:\$PATH

echo "Conda default env: \$CONDA_DEFAULT_ENV"
which python
python --version

srun python /ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/embed_flux.py \\
    --dataset ${dataset_str} \\
    --model_cfgs ${configs[@]} \\
    --model_weights ${weights[@]} \\
    --output_path ${output_path} \\
    --batch_size 256 \\
    --num_workers 5 \\
    --channel 2 \\
    --use_data_parallel \\
    --auto_scale_batch \\
    --max_samples -1 \\
    --seed 0
EOF