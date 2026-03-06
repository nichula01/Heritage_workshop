#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_registry import list_datasets, get_dataset_info

print("Available datasets:")
for name in list_datasets():
    info = get_dataset_info(name)
    print(f"- {info.name}")
    print(f"  task: {info.task}")
    print(f"  modality: {info.modality}")
    print(f"  annotation_type: {info.annotation_type}")
    print(f"  root_hint: {info.root_hint}")
    print(f"  notes: {info.notes}")
    print()
