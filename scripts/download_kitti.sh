#!/usr/bin/env bash
# KITTI raw drive used by the notebooks (public S3 mirror of the official download).
#   scripts/download_kitti.sh [drive]     default: 2011_09_26_drive_0001
set -euo pipefail
cd "$(dirname "$0")/.."
DRIVE="${1:-2011_09_26_drive_0001}"
DATE="${DRIVE:0:10}"
S3=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
mkdir -p data/raw && cd data/raw
curl -L --retry 5 -C - -o "${DATE}_calib.zip" "$S3/${DATE}_calib.zip"
curl -L --retry 5 -C - -o "${DRIVE}_sync.zip" "$S3/${DRIVE}/${DRIVE}_sync.zip"
unzip -q -o "${DATE}_calib.zip" && unzip -q -o "${DRIVE}_sync.zip" && rm "${DATE}_calib.zip" "${DRIVE}_sync.zip"
echo "ready: data/raw/${DATE}/${DRIVE}_sync"
