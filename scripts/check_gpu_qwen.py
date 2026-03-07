#!/usr/bin/env python3
import torch

print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_name:", torch.cuda.get_device_name(0))
    if hasattr(torch.cuda, "is_bf16_supported"):
        print("bf16_supported:", torch.cuda.is_bf16_supported())
