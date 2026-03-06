import json
import random
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
