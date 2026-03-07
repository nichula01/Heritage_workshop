#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment_minicpm.yml || conda env update -f environment_minicpm.yml --prune

echo
echo "[INFO] Activate with:"
echo "conda activate heritage_minicpm"

echo
echo "[INFO] Verifying installation..."
conda run -n heritage_minicpm python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY
