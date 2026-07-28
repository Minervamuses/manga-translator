#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models"
MODEL_PATH="${MODEL_DIR}/comictextdetector.pt"
MODEL_URL="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"

mkdir -p "${MODEL_DIR}"

echo "Downloading comictextdetector.pt ..."
wget -O "${MODEL_PATH}" "${MODEL_URL}"
echo "Done. Model saved to ${MODEL_PATH}"
