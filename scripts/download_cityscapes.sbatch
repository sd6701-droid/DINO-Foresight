#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600     # send USR1 to the batch shell 10 min before the time limit
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=cityscapes_download
#SBATCH --output=/scratch/sd6701/DINO-Foresight/logs/cityscapes_download/cityscapes_download_%j.log
#SBATCH --error=/scratch/sd6701/DINO-Foresight/logs/cityscapes_download/cityscapes_download_%j.err
#SBATCH --requeue                # restart after preemption/crash; the script is idempotent
#SBATCH --open-mode=append       # requeued attempt appends to the SAME %j log
#
# Downloads and extracts the Cityscapes packages needed by DINO-Foresight.
# One-time setup on the login node (so the job can log in unattended):
#   pip install cityscapesscripts
#   mkdir -p ~/.local/share/cityscapesscripts
#   echo '{"username": "YOUR_EMAIL", "password": "YOUR_PASSWORD"}' > ~/.local/share/cityscapesscripts/credentials.json
#   chmod 600 ~/.local/share/cityscapesscripts/credentials.json
# Then:  mkdir -p /scratch/sd6701/DINO-Foresight/logs/cityscapes_download
#        sbatch scripts/download_cityscapes.sbatch                 # cpu_short (header default)
#        sbatch --partition=cs scripts/download_cityscapes.sbatch  # second copy on cs
# Whichever copy starts first cancels the other pending copy automatically.
#
# Every stage is resumable, so cancelling and resubmitting is always safe:
#   download  - csDownload -r continues a partial zip via HTTP Range requests and
#               verifies the MD5; a finished package is marked and never re-fetched
#   extract   - scripts/unzip_resume.py skips files already on disk with the right
#               size and re-extracts only the one that was cut off
#   labels    - csCreateTrainIdLabelImgs is skipped once its marker exists
# The job also requeues itself shortly before the partition time limit, so one
# sbatch is enough even though the 324 GB package takes longer than one slot.
# Check the limit with: sinfo -p cpu_short -o "%P %l"  and lower --time if needed.
# Data goes to /scratch/$USER/datasets/cityscapes (override with DATASETS_ROOT or
# CITYSCAPES_DEST). Before downloading it deletes the unrelated scand/huron/recon
# datasets from DATASETS_ROOT (override with REMOVE_DATASETS="..." or disable with
# REMOVE_DATASETS="").

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# If the partition's time limit is about to hit, requeue ourselves. The job is
# idempotent, so the next attempt simply picks up where this one stopped.
requeue_self() {
    echo "[$(date)] Approaching time limit, requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    trap requeue_self USR1
fi

DATASETS_ROOT=${DATASETS_ROOT:-/scratch/$USER/datasets}
DEST=${CITYSCAPES_DEST:-$DATASETS_ROOT/cityscapes}
mkdir -p "$DATASETS_ROOT" "$DEST"
echo "[$(date)] host=$(hostname) partition=${SLURM_JOB_PARTITION:-none} datasets_root=$DATASETS_ROOT dest=$DEST"

# ---------------------------------------------------------------------------
# Submitted to several partitions? The first copy that starts cancels every
# still-pending copy with the same job name, and takes an exclusive lock so a
# second copy that starts at the same instant exits instead of writing to the
# same zip files.
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    pending=$(squeue -h -u "$USER" --name="$SLURM_JOB_NAME" --states=PENDING -o "%i" | grep -vx "$SLURM_JOB_ID" || true)
    if [[ -n "$pending" ]]; then
        echo "[$(date)] Job $SLURM_JOB_ID started on $SLURM_JOB_PARTITION; cancelling pending twins: $(echo $pending)"
        scancel $pending || true
    fi
fi
exec 9>"$DEST/.download.lock"
if ! flock -n 9; then
    echo "[$(date)] Another download job already holds $DEST/.download.lock; exiting"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 0: free space by removing unrelated navigation-trajectory datasets.
# Only the exact directory names listed here are removed, and only if they
# are direct children of $DATASETS_ROOT. Set REMOVE_DATASETS="" to skip.
# ---------------------------------------------------------------------------
REMOVE_DATASETS=${REMOVE_DATASETS-"scand huron recon"}
for name in $REMOVE_DATASETS; do
    target="$DATASETS_ROOT/$name"
    if [[ -d "$target" && ! -L "$target" ]]; then
        echo "[$(date)] Removing $target ($(du -sh "$target" 2>/dev/null | cut -f1))"
        rm -rf -- "$target"
    else
        echo "[$(date)] Skip $target (not found)"
    fi
done
echo "[$(date)] Free space in $DATASETS_ROOT:"
df -h "$DATASETS_ROOT" || true

# ---------------------------------------------------------------------------
# Step 1: download. One package at a time so each gets its own done-marker.
# ---------------------------------------------------------------------------
mkdir -p "$DEST"
cd "$DEST"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
command -v csDownload >/dev/null || { echo "csDownload not found: pip install cityscapesscripts"; exit 1; }

PACKAGES=(gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip leftImg8bit_sequence_trainvaltest.zip)

for pkg in "${PACKAGES[@]}"; do
    marker=".${pkg}.downloaded"
    if [[ -f "$marker" ]]; then
        echo "[$(date)] skip download of $pkg (complete, MD5 verified earlier)"
        continue
    fi
    echo "[$(date)] Downloading $pkg into $DEST"
    # Run in the background and wait so the USR1 trap can fire mid-download.
    set +e
    csDownload -r -d "$DEST" "$pkg" > "$pkg.log" 2>&1 &
    wait $!
    rc=$?
    set -e
    tail -n 3 "$pkg.log"
    if [[ $rc -ne 0 ]]; then
        if grep -q "MD5 sum of downloaded file does not match" "$pkg.log"; then
            echo "[$(date)] $pkg is corrupt, deleting so the next attempt re-downloads it"
            rm -f -- "$pkg"
        else
            echo "[$(date)] download of $pkg interrupted (rc=$rc); partial file kept for resume"
        fi
        exit $rc
    fi
    touch "$marker"
    echo "[$(date)] $pkg downloaded and MD5 verified"
done

# ---------------------------------------------------------------------------
# Step 2: resumable extraction.
# ---------------------------------------------------------------------------
for pkg in "${PACKAGES[@]}"; do
    marker=".${pkg}.extracted"
    if [[ -f "$marker" ]]; then
        echo "skip extraction of $pkg (already extracted)"
        continue
    fi
    echo "[$(date)] Extracting $pkg"
    python "$REPO_DIR/scripts/unzip_resume.py" "$pkg" "$DEST" --exclude README license.txt &
    wait $!
    touch "$marker"
    echo "[$(date)] extracted $pkg"
done

# ---------------------------------------------------------------------------
# Step 3: 19-class labelTrainIds for gtFine (needed by Downstream/data.py).
# ---------------------------------------------------------------------------
if [[ -f .labelTrainIds.done ]]; then
    echo "skip labelTrainIds (already generated)"
else
    echo "[$(date)] Creating labelTrainIds for gtFine"
    CITYSCAPES_DATASET="$DEST" csCreateTrainIdLabelImgs &
    wait $!
    touch .labelTrainIds.done
fi

echo "[$(date)] Done. Layout:"
ls "$DEST"
du -sh "$DEST"/leftImg8bit "$DEST"/gtFine "$DEST"/leftImg8bit_sequence
echo "Zips can now be removed to free ~340 GB:  rm $DEST/*.zip"
