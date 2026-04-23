#!/bin/bash
#SBATCH --job-name=meshtusk_train
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source /cephfs/store/damtp/gam37/venv-patch-modulator/bin/activate

# Move to project directory
cd /cephfs/store/damtp/gam37/patch-modulator_dev

# Run training script
python controllable_patching_striding/train.py \
data=TRL_2D \
data.well_base_path=./datasets \
trainer.max_epoch=2 \
trainer.max_rollout_steps=10 \
trainer.prediction_type=delta \
trainer.video_validation=False \
trainer.image_validation=False \
data.module_parameters.batch_size=2 \
data.module_parameters.max_samples=100 \
data_workers=4 \
optimizer.lr=0.0001 \
model.hidden_dim=192 \
model.groups=12 \
model.processor_blocks=12 \
model.drop_path=0.1 \
model/processor/space_mixing=full_spatial_attention \
model.processor.space_mixing.num_heads=3 \
model.processor.time_mixing.num_heads=3 \
model.causal_in_time=True \
model.jitter_patches=False \
model/encoder=vstride_encoder \
model.encoder.learned_pad=True \
model.encoder.variable_deterministic_ds=False \
model.encoder.base_kernel_size2d="[[4,4],[4,4]]" \
model.encoder.kernel_scales_seq="[[4,4]]" \
model/decoder=vstride_decoder \
model.decoder.learned_pad=True \
model.decoder.base_kernel_size2d="[[4,4],[4,4]]" \
model.infer="[4,4]" \
model.twod_only=True \
model.threed_only=False \
logger=wandb \
logger.wandb_project_name=patch-modulator-training \
logger.wandb_entity=MeshTusk \
auto_resume=False
