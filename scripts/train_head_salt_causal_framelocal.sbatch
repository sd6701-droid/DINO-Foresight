#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=head_causal_framelocal
#SBATCH --output=/scratch/sd6701/DINO-Foresight/logs/head_causal_framelocal/head_causal_framelocal_%j.log
#SBATCH --error=/scratch/sd6701/DINO-Foresight/logs/head_causal_framelocal/head_causal_framelocal_%j.err

set -euo pipefail

REPO_DIR=/scratch/sd6701/DINO-Foresight
DATA_DIR=/scratch/sd6701/datasets/cityscapes
CKPT=/scratch/sd6701/salt/p-salt/results/student_ssv2_50ep/framelocal_T_bidir_decbidir/checkpoints/ece40e2a/last/checkpoint.pth
# One short tag per encoder checkpoint. It names the head's output dir and the wandb run,
# and the predictor sbatch must use the same CKPT + ENC_TAG to find this head.
ENC_TAG=salt-framelocal-tbidir
LOG_DIR=/scratch/sd6701/DINO-Foresight/logs/head_causal_framelocal
DST_DIR=/scratch/sd6701/DINO-Foresight/results/head_causal_framelocal/$ENC_TAG

mkdir -p "$LOG_DIR" "$DST_DIR"

echo "[$(date)] host=$(hostname) gpus=$SLURM_GPUS_ON_NODE job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
# activate.d/env_vars.sh expands $LD_LIBRARY_PATH, which is fatal under set -u
set +u
conda activate salt
set -u

cd "$REPO_DIR/Downstream"

python train_dpt_head.py \
    --wandb_project   dino-foresight \
    --wandb_run_name  head-${ENC_TAG}-10ep \
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
    --num_gpus        1 \
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

echo "Head checkpoint(s) under: $DST_DIR"
find "$DST_DIR" -name "last.ckpt" -path "*/checkpoints/*"
echo "[$(date)] Done"
