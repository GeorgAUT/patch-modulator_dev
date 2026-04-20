#!/bin/bash
#SBATCH --job-name=meshtusk_train
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=23:00:00
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
server=local distribution=local \
data=TRL_2D \
model=isotropic_model_small \
data.well_base_path=./datasets \
trainer.max_epoch=300 \
trainer.video_validation=False \
trainer.image_validation=False \
data.module_parameters.max_samples=2000 \
auto_resume=False \
logger=wandb \
logger.wandb_project_name=patch-modulator-training \
logger.wandb_entity=MeshTusk
