#!/usr/bin/env python3
from datetime import datetime

from src.dataset_registry import get_dataset_info
from src.utils import ensure_dir, save_json, set_seed


def main():
    set_seed(42)

    dataset_name = "artpedia"
    model_name = "placeholder_vlm_baseline"
    run_name = "baseline_run"

    dataset = get_dataset_info(dataset_name)
    out_dir = ensure_dir(f"results/{run_name}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "project": "Heritage_workshop",
        "run_name": run_name,
        "dataset": dataset.name,
        "task": dataset.task,
        "model": model_name,
        "status": "scaffold_only",
        "next_action": "Replace this script with real inference/evaluation once dataset is finalized."
    }

    save_json(result, out_dir / "summary.json")
    print(f"Saved scaffold result to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
