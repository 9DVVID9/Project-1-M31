#!/bin/bash
# === daily_zip.sh ===
# Compress yesterday's images into a zip and store it in /home/david_iancu999/traffic_arago/traffic_arago_starter/zips

BASE_DIR="/home/david_iancu999/traffic_arago/traffic_arago_starter"
ZIP_DIR="$BASE_DIR/zips"
YESTERDAY_UTC=$(date -u -d "yesterday" +%Y/%m/%d)

mkdir -p "$ZIP_DIR"

ZIP_NAME="images_$(date -u -d 'yesterday' +%Y%m%d).zip"
ZIP_PATH="$ZIP_DIR/$ZIP_NAME"
SRC_DIR="$BASE_DIR/data/raw/images/$YESTERDAY_UTC"

if [ -d "$SRC_DIR" ]; then
    echo "Zipping $SRC_DIR -> $ZIP_PATH"
    zip -r "$ZIP_PATH" "$SRC_DIR"
else
    echo "No images found for $YESTERDAY_UTC, skipping."
fi
