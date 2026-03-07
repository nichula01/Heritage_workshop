#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.qwen_vlm import QwenVLM

CSV_PATH = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["download_status"].isin(["ok", "exists"])].head(3).copy()

    model = QwenVLM("Qwen/Qwen2.5-VL-3B-Instruct")

    for i, row in df.iterrows():
        print("\n" + "=" * 100)
        print("QUESTION:", row["question"])
        print("GOLD:", row["short_answer"])
        print("IMAGE:", row["local_image_path"])
        pred = model.answer_image_only(
            image_path=str(row["local_image_path"]),
            question=str(row["question"]),
            max_new_tokens=32
        )
        print("PRED:", pred)


if __name__ == "__main__":
    main()
