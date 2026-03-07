#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Upgrading pip"
pip install -U pip

echo "[INFO] Installing torch stack if missing"
python - <<'PY'
import importlib.util
print("torch_installed:", importlib.util.find_spec("torch") is not None)
PY

if ! python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
  pip install torch torchvision torchaudio
fi

echo "[INFO] Installing model requirements"
pip install -r requirements-models.txt

echo "[INFO] Installing latest transformers from source"
pip install "git+https://github.com/huggingface/transformers"

echo
echo "[INFO] Verifying setup"
python - <<'PY'
import importlib.util

mods = ["torch", "transformers", "qwen_vl_utils", "accelerate"]
for m in mods:
    print(f"{m}:", importlib.util.find_spec(m) is not None)

import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY
