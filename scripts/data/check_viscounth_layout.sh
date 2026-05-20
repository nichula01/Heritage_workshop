#!/usr/bin/env bash
set -euo pipefail

DST="data/raw/viscounth_repo"

echo "[INFO] Checking path: $DST"
if [ ! -d "$DST" ]; then
  echo "[ERROR] Folder not found: $DST"
  exit 1
fi

echo
echo "================ TOP LEVEL ================"
find "$DST" -maxdepth 1 | sort

echo
echo "================ DEPTH 2 ================"
find "$DST" -maxdepth 2 | sort

echo
echo "================ DATASET 2.0 ================"
if [ -d "$DST/Dataset 2.0" ]; then
  find "$DST/Dataset 2.0" -maxdepth 3 | sort
else
  echo "[WARN] Dataset 2.0 not found"
fi

echo
echo "================ DESRIPTION ================"
if [ -d "$DST/Desription" ]; then
  find "$DST/Desription" -maxdepth 3 | sort
else
  echo "[WARN] Desription not found"
fi

echo
echo "================ EXCEL FILES ================"
find "$DST" -maxdepth 2 \( -iname "*.xlsx" -o -iname "*.xls" -o -iname "*.csv" \) | sort

echo
echo "[OK] Layout check complete"
