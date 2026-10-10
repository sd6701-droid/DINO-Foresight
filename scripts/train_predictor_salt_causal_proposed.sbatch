#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=pred_causal_proposed
#SBATCH --output=/scratch/sd6701/DINO-Foresight/logs/pred_causal_proposed/pred_causal_proposed_%j.log
#SBATCH --error=/scratch/sd6701/DINO-Foresight/logs/pred_causal_proposed/pred_causal_proposed_%j.err

set -euo pipefail

REPO_DIR=/scratch/sd6701/DINO-Foresight
DATA_DIR=/scratch/sd6701/datasets/cityscapes/leftImg8bit_sequence
CKPT=/scratch/sd6701/vjepa2_runs/vits.ssv2.256px.16f.1000ep/pretrain-256px-16f-1000ep_20260927-161429/best.pt::target_encoder
# Same ENC_TAG as train_head_salt_causal_proposed.sbatch (used for the wandb run name).
ENC_TAG=vjepa2-target
LOG_DIR=/scratch/sd6701/DINO-Foresight/logs/pred_causal_proposed
DST_DIR=/scratch/sd6701/DINO-Foresight/results/predictor_causal_proposed
HEAD_ROOT=/scratch/sd6701/DINO-Foresight/results/head_causal_proposed

mkdir -p "$LOG_DIR" "$DST_DIR"

echo "[$(date)] host=$(hostname) gpus=$SLURM_GPUS_ON_NODE job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
# activate.d/env_vars.sh expands $LD_LIBRARY_PATH, which is fatal under set -u
set +u
conda activate salt
set -u

# Heads live under $HEAD_ROOT/[<ENC_TAG>/]dino-foresight/<random wandb run id>/checkpoints/last.ckpt
# (Lightning puts ModelCheckpoint under the WandbLogger save_dir). Older heads sit directly under
# $HEAD_ROOT, newer ones under $HEAD_ROOT/$ENC_TAG; search both. Among all candidates (newest
# first) take the first whose saved hparams `student_ckpt` equals this predictor's CKPT, so a head
# trained on a different encoder can never be picked silently.
if [ ! -d "$HEAD_ROOT" ]; then
    echo "ERROR: $HEAD_ROOT does not exist. Run train_head_salt_causal_proposed.sbatch first." >&2
    exit 1
fi
HEAD_CANDS=()
while IFS= read -r c; do HEAD_CANDS+=("$c"); done < <(find "$HEAD_ROOT" -name "last.ckpt" -path "*/checkpoints/*" -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
HEAD_CKPT=$(python - "$CKPT" "${HEAD_CANDS[@]:-}" <<'EOF'
import sys, torch
from argparse import Namespace
enc_ckpt = sys.argv[1]
cands = [c for c in sys.argv[2:] if c]
seen = []
for c in cands:
    hp = torch.load(c, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    hp = hp.get("args", hp)
    hp = vars(hp) if isinstance(hp, Namespace) else dict(hp)
    head_enc = hp.get("student_ckpt")
    seen.append((c, head_enc))
    if head_enc == enc_ckpt:
        print(c)
        break
else:
    msg = "\n".join(f"  {c}  (trained on: {e})" for c, e in seen) or "  (none found)"
    sys.exit(f"ERROR: no head under HEAD_ROOT was trained on\n  {enc_ckpt}\nCandidates:\n{msg}")
EOF
)
if [ -z "$HEAD_CKPT" ]; then
    echo "ERROR: head selection failed" >&2
    exit 1
fi
echo "Using head checkpoint (encoder verified): $HEAD_CKPT"

cd "$REPO_DIR"

python train.py \
    --wandb_project    dino-foresight \
    --wandb_run_name   predictor-${ENC_TAG}-10ep \
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
