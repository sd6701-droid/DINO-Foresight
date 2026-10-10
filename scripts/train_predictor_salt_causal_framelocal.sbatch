#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=pred_causal_framelocal
#SBATCH --output=/scratch/sd6701/DINO-Foresight/logs/pred_causal_framelocal/pred_causal_framelocal_%j.log
#SBATCH --error=/scratch/sd6701/DINO-Foresight/logs/pred_causal_framelocal/pred_causal_framelocal_%j.err

set -euo pipefail

REPO_DIR=/scratch/sd6701/DINO-Foresight
DATA_DIR=/scratch/sd6701/datasets/cityscapes/leftImg8bit_sequence
CKPT=/scratch/sd6701/salt/p-salt/results/student_ssv2_50ep/framelocal_T_bidir_decbidir/checkpoints/ece40e2a/last/checkpoint.pth
LOG_DIR=/scratch/sd6701/DINO-Foresight/logs/pred_causal_framelocal
DST_DIR=/scratch/sd6701/DINO-Foresight/results/predictor_causal_framelocal

HEAD_CKPT=$(find /scratch/sd6701/DINO-Foresight/results/head_causal_framelocal -name "last.ckpt" -path "*/checkpoints/*" | head -1)
if [ -z "$HEAD_CKPT" ]; then
    echo "ERROR: No head checkpoint found in results/head_causal_framelocal" >&2
    exit 1
fi
echo "Using head checkpoint: $HEAD_CKPT"

mkdir -p "$LOG_DIR" "$DST_DIR"

echo "[$(date)] host=$(hostname) gpus=$SLURM_GPUS_ON_NODE job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
# activate.d/env_vars.sh expands $LD_LIBRARY_PATH, which is fatal under set -u
set +u
conda activate salt
set -u

cd "$REPO_DIR"

python train.py \
    --wandb_project    dino-foresight \
    --wandb_run_name   predictor-salt-causal-framelocal-10ep \
    --data_path        "$DATA_DIR" \
    --feature_extractor student \
    --student_ckpt     "$CKPT" \
    --student_arch     salt_vit_small_rope \
    --img_size         448,896 \
    --d_layers         2,5,8,11 \
    --sequence_length  5 \
    --step             1 \
    --batch_size       2 \
    --num_gpus         1 \
    --max_epochs       10 \
    --eval_freq        1 \
    --eval_modality    segm \
    --eval_mode_during_training \
    --head_ckpt        "$HEAD_CKPT" \
    --num_classes      19 \
    --dpt_out_channels 128,256,512,512 \
    --use_bn \
    --hidden_dim       384 \
    --heads            6 \
    --layers           8 \
    --precision        16-mixed \
    --num_workers      8 \
    --evaluate \
    --dst_path         "$DST_DIR" \
    --seed             123

echo "[$(date)] Done"
