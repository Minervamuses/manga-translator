#!/usr/bin/env bash
set -euo pipefail

FONT_DIR="fonts"
FONT_PATH="${FONT_DIR}/Iansui-Regular.ttf"
FONT_ZIP="/tmp/iansui.zip"
EXTRACT_DIR="/tmp/iansui_extract"
FONT_URL="https://github.com/ButTaiwan/iansui/releases/download/v1.020/iansui.zip"

mkdir -p "${FONT_DIR}"

echo "Downloading Iansui font..."
wget -O "${FONT_ZIP}" "${FONT_URL}"

rm -rf "${EXTRACT_DIR}"
unzip -o "${FONT_ZIP}" -d "${EXTRACT_DIR}" >/dev/null

SOURCE_FONT="$(find "${EXTRACT_DIR}" -type f -name "Iansui-Regular.ttf" | head -n 1)"
if [[ -z "${SOURCE_FONT}" ]]; then
  echo "Iansui-Regular.ttf not found after unzip."
  exit 1
fi

cp "${SOURCE_FONT}" "${FONT_PATH}"
rm -rf "${FONT_ZIP}" "${EXTRACT_DIR}"

echo "Done. Font saved to ${FONT_PATH}"
