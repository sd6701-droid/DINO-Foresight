#!/bin/bash
# Submit Stage 1 (DPT head) and Stage 2 (predictor) for the T_bidir student checkpoint as a
# Slurm dependency chain. Stage 2 starts only if Stage 1 exits 0 (afterok), is cancelled if
# Stage 1 fails (kill-on-invalid-dep), and is told the Stage 1 job id so it can verify that
# the head it loads was freshly produced by that job (see EXPECT_HEAD_JOB in
# train_predictor_tbidir.sbatch), on top of the encoder-hparams and pointer checks.
#
# Usage:  bash scripts/run_tbidir_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

HEAD_JOB=$(sbatch --parsable train_head_tbidir.sbatch)
HEAD_JOB=${HEAD_JOB%%;*}   # --parsable may append ";cluster"
echo "Stage 1 (head)      submitted: job $HEAD_JOB"

PRED_JOB=$(sbatch --parsable \
    --dependency=afterok:"$HEAD_JOB" \
    --kill-on-invalid-dep=yes \
    --export=ALL,EXPECT_HEAD_JOB="$HEAD_JOB" \
    train_predictor_tbidir.sbatch)
PRED_JOB=${PRED_JOB%%;*}
echo "Stage 2 (predictor) submitted: job $PRED_JOB  (afterok:$HEAD_JOB, EXPECT_HEAD_JOB=$HEAD_JOB)"

echo
echo "Logs:"
echo "  head:      /scratch/sd6701/DINO-Foresight/logs/head_tbidir/head_tbidir_${HEAD_JOB}.log"
echo "  predictor: /scratch/sd6701/DINO-Foresight/logs/pred_tbidir/pred_tbidir_${PRED_JOB}.log"
echo
echo "In the predictor log, expect these lines before training starts:"
echo "  Using head checkpoint (encoder verified): ..."
echo "  Pointer check OK: ..."
echo "  Fresh-head check OK: head produced by Stage 1 job $HEAD_JOB (this pipeline)"
echo
squeue -u "$USER" -j "$HEAD_JOB,$PRED_JOB" -o "%.10i %.14j %.8T %.20E" || true
