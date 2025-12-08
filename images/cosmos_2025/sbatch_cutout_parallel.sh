#!/bin/bash
#SBATCH --time=20:00:00

#SBATCH --job-name=cutout-par
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120000
#SBATCH --output=cutout_parallel_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yacheng@mpia.de


module purge
module load anaconda/3/2023.03

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

source ~/.bashrc
conda activate astrodino

WORKDIR="/ptmp/yacheng/outthere_ssl/images/cosmos_2025"
cd "$WORKDIR"

OUTPUT_DIR="outputs/"
mkdir -p "$OUTPUT_DIR"

MAX_WORKERS=${SLURM_CPUS_PER_TASK:-20}

srun python cutout_export_parallel.py \
    --catalog COSMOSWeb_mastercatalog_v1_photom_primary.fits \
    --filters f115w f150w \
    --output-dir "$OUTPUT_DIR" \
    --base-dir . \
    --segmentation-dir segmentation_maps \
    --chunk-size 256 \
    --image-size 128 \
    --seg-size 128 \
    --max-workers "$MAX_WORKERS"
