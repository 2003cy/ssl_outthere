#!/bin/bash

# define config_file list (three different configurations)
configs=(

)

# define pretrained_weights list (three different weights, corresponding to the above configurations)
weights=(


)

# loop through and submit three jobs
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    weight="${weights[$i]}"
    echo "Submitting job $((i+1)) with config: $config and weight: $weight"
    
    # submit job, modifying --config_file and --pretrained_weights parameters
    sbatch <<EOF
#!/bin/bash -l
#SBATCH -t 00:06:00
#SBATCH --nodes=1
#SBATCH --partition=gpudev  # Use the GPU development partition
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000
#SBATCH --output=/ptmp/yacheng/outthere_ssl/compute_embedding-%j.log
#SBATCH --mail-type=none
#SBATCH --mail-user=yacheng@mpia.de

module purge
module load anaconda/3/2023.03
module load cuda/12.6

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

conda activate astrodino

srun python /ptmp/yacheng/outthere_ssl/compute_embeddings.py --config_file $config --pretrained_weights $weight --dataset "LegacySurvey:split=val:root=/ptmp/yacheng/outthere_ssl/:extra=\"\"" --batch_size 256
EOF
done