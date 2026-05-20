#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="heritage_workshop"

echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda env create -f environment.yml || conda env update -f environment.yml --prune

echo
echo "[INFO] Activate it with:"
echo "conda activate ${ENV_NAME}"
echo
echo "[INFO] Verifying core packages..."
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
import pandas
import numpy
import sklearn
import yaml
import requests
from PIL import Image

print("Python:", sys.version)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("sklearn:", sklearn.__version__)
print("yaml: OK")
print("requests: OK")
print("PIL: OK")
PY
