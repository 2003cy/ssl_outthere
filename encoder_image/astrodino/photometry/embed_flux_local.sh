#!/bin/bash
# Simple wrapper to run embed_flux.py directly from an interactive shell (no sbatch).
# Expect that the conda env / CUDA modules are already activated before calling.
#!/bin/bash
LOG_FILE="/ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/embed_flux_local.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source /mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2023.03/etc/profile.d/conda.sh

source ~/.bashrc

conda activate astrodino

export PATH=$CONDA_PREFIX/bin:$PATH

# Debug: Check which conda and python
echo "Conda default env: $CONDA_DEFAULT_ENV"
which python
python --version

configs=(
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable/config.yaml'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable_multi_epochs/config.yaml'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable_multi_epochs_vit_large/config.yaml'
)

weights=(
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable/eval/manual_18000/teacher_checkpoint.pth'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable_multi_epochs/eval/manual_60000/teacher_checkpoint.pth'
    '/ptmp/yacheng/outthere_ssl/AstroCLIP/outputs/astroclip_image/astrodino_3z_band_stable_multi_epochs_vit_large/eval/training_23999/teacher_checkpoint.pth'
)

dataset_str="LegacySurvey:split=test:root=/ptmp/yacheng/outthere_ssl/:extra=\"\""
output_path='/ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/legacy_flux_embeddings_local.h5'

if [[ ${#configs[@]} -ne ${#weights[@]} ]]; then
    echo "configs and weights length mismatch" >&2
    exit 1
fi

echo "Running embed_flux.py for ${#configs[@]} models (interactive run)"

conda run -n astrodino python /ptmp/yacheng/outthere_ssl/AstroCLIP/astroclip/astrodino/photometry/embed_flux.py \
    --dataset "${dataset_str}" \
    --model_cfgs "${configs[@]}" \
    --model_weights "${weights[@]}" \
    --output_path "${output_path}" \
    --batch_size 256 \
    --num_workers 4 \
    --channel 2 \
    --use_data_parallel \
    --auto_scale_batch \
    --max_samples 10 \
    --seed 0
