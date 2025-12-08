#!/bin/bash
#SBATCH -J crossmatch
#SBATCH -p small
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /ptmp/yacheng/outthere_ssl/train_3z_band-22839280.log/crossmatch_%j.out


module purge
module load anaconda/3/2023.03
module load cuda/12.6
#module load gcc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

conda activate astrodino

# 运行脚本（参数均使用默认，可按需覆盖）
python astroclip/downstream_tasks/morphology_classification/morphology_utils/cross_match.py \
  --root_dir /ptmp/yacheng/outthere_ssl/images \
  --save_path /ptmp/yacheng/outthere_ssl/images/galaxy_zoo/gz5_decals_crossmatched.hdf5 \
  --survey_path /ptmp/yacheng/outthere_ssl/images/galaxy_zoo/gz_decals_volunteers_5.csv