#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Submitting causal_framelocal pipeline ==="
JOB_FL_HEAD=$(sbatch --parsable "$SCRIPT_DIR/train_head_salt_causal_framelocal.sbatch")
echo "Stage 1 (head) submitted: job $JOB_FL_HEAD"
JOB_FL_PRED=$(sbatch --parsable --dependency=afterok:$JOB_FL_HEAD "$SCRIPT_DIR/train_predictor_salt_causal_framelocal.sbatch")
echo "Stage 2 (predictor) submitted: job $JOB_FL_PRED (starts after $JOB_FL_HEAD)"

echo ""
echo "=== Submitting causal_proposed pipeline ==="
JOB_CP_HEAD=$(sbatch --parsable "$SCRIPT_DIR/train_head_salt_causal_proposed.sbatch")
echo "Stage 1 (head) submitted: job $JOB_CP_HEAD"
JOB_CP_PRED=$(sbatch --parsable --dependency=afterok:$JOB_CP_HEAD "$SCRIPT_DIR/train_predictor_salt_causal_proposed.sbatch")
echo "Stage 2 (predictor) submitted: job $JOB_CP_PRED (starts after $JOB_CP_HEAD)"

echo ""
echo "=== All jobs submitted ==="
echo "causal_framelocal: head=$JOB_FL_HEAD -> predictor=$JOB_FL_PRED"
echo "causal_proposed:   head=$JOB_CP_HEAD -> predictor=$JOB_CP_PRED"
echo "Monitor with: squeue -u \$USER"
