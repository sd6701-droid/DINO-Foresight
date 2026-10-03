#!/bin/bash
#SBATCH --job-name=cityscapes_dl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
# Add your CPU partition if required, e.g.:
# #SBATCH --partition=cpu
#
# Downloads and extracts the Cityscapes packages needed by DINO-Foresight.
# One-time setup on the login node (so the job can log in unattended):
#   pip install cityscapesscripts
#   mkdir -p ~/.local/share/cityscapesscripts
#   echo '{"username": "YOUR_EMAIL", "password": "YOUR_PASSWORD"}' > ~/.local/share/cityscapesscripts/credentials.json
#   chmod 600 ~/.local/share/cityscapesscripts/credentials.json
# Then:  sbatch scripts/download_cityscapes.sbatch
# Resubmitting is safe: downloads resume and extracted packages are skipped.

set -euo pipefail

DEST=${CITYSCAPES_DEST:-~/datasets/cityscapes}
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
