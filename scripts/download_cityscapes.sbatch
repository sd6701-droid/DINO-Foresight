#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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
#        sbatch scripts/download_cityscapes.sbatch
# Resubmitting is safe: downloads resume and extracted packages are skipped.
# Before downloading it deletes the unrelated scand/huron/recon datasets from
# ~/datasets (override with REMOVE_DATASETS="..." or disable with REMOVE_DATASETS="").

set -euo pipefail

DATASETS_ROOT=${DATASETS_ROOT:-~/datasets}
DEST=${CITYSCAPES_DEST:-$DATASETS_ROOT/cityscapes}

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
df -h "$DATASETS_ROOT"

# ---------------------------------------------------------------------------
# Step 1: download, verify and extract Cityscapes.
# ---------------------------------------------------------------------------
mkdir -p "$DEST"
cd "$DEST"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

PACKAGES=(gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip leftImg8bit_sequence_trainvaltest.zip)

echo "[$(date)] Downloading into $DEST"
csDownload -r -d "$DEST" "${PACKAGES[@]}"

echo "[$(date)] Verifying checksums"
for pkg in "${PACKAGES[@]}"; do
    if [[ -f "${pkg}.md5" ]]; then
        md5sum -c "${pkg}.md5"
    fi
done

echo "[$(date)] Extracting"
for pkg in "${PACKAGES[@]}"; do
    marker=".${pkg}.extracted"
    if [[ -f "$marker" ]]; then
        echo "skip $pkg (already extracted)"
        continue
    fi
    unzip -q -o "$pkg" -x "README" "license.txt" -d "$DEST"
    touch "$marker"
    echo "[$(date)] extracted $pkg"
done

echo "[$(date)] Creating 19-class labelTrainIds for gtFine"
CITYSCAPES_DATASET="$DEST" csCreateTrainIdLabelImgs

echo "[$(date)] Done. Layout:"
ls "$DEST"
du -sh "$DEST"/leftImg8bit "$DEST"/gtFine "$DEST"/leftImg8bit_sequence
