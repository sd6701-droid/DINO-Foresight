#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=head_salt_vanilla
#SBATCH --output=/scratch/sd6701/DINO-Foresight/logs/head_salt_vanilla/head_salt_vanilla_%j.log
#SBATCH --error=/scratch/sd6701/DINO-Foresight/logs/head_salt_vanilla/head_salt_vanilla_%j.err
#
# Train a DPT segmentation head on frozen vanilla SALT ViT-S/16 features
# (ceiling check: single Cityscapes images, not sequences).
# 10 epochs, 2 H100s, 16-mixed precision.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR=/scratch/sd6701/DINO-Foresight
DATA_DIR=/scratch/sd6701/datasets/cityscapes
CKPT=/scratch/sd6701/salt/p-salt/results/student_ssv2_50ep/salt/checkpoints/94f03805/last/checkpoint.pth
LOG_DIR=/scratch/sd6701/DINO-Foresight/logs/head_salt_vanilla
DST_DIR=/scratch/sd6701/DINO-Foresight/results/head_salt_vanilla

mkdir -p "$LOG_DIR" "$DST_DIR"

echo "[$(date)] host=$(hostname) gpus=$SLURM_GPUS_ON_NODE job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Environment ──────────────────────────────────────────────────────────────
source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
# activate.d/env_vars.sh expands $LD_LIBRARY_PATH, which is fatal under set -u
set +u
conda activate salt
set -u

# Wandb — set your API key here or run `wandb login` once on the cluster
# export WANDB_API_KEY="your-key-here"

# ── Run ──────────────────────────────────────────────────────────────────────
cd "$REPO_DIR/Downstream"

python train_dpt_head.py \
    --wandb_project   dino-foresight \
    --wandb_run_name  head-salt-vanilla-10ep \
    --data_path       "$DATA_DIR" \
    --feature_extractor student \
    --student_ckpt    "$CKPT" \
    --student_arch    salt_vit_small_rope \
    --modality        segm \
    --num_classes     19 \
    --img_size        448,896 \
    --dlayers         2,5,8,11 \
    --dpt_out_channels 128,256,512,512 \
    --use_bn \
    --nfeats          256 \
    --batch_size      16 \
    --num_gpus        2 \
    --max_epochs      10 \
    --eval_freq       1 \
    --lr_base         1e-4 \
    --weight_decay    1e-4 \
    --optimizer       adamw \
    --scheduler       poly \
    --warmup_p        0.05 \
    --precision       16-mixed \
    --num_workers     8 \
    --num_workers_val 8 \
    --dst_path        "$DST_DIR" \
    --seed            123

echo "[$(date)] Done"
