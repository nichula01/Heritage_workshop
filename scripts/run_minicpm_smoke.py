#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.minicpm_vlm import MiniCPMVLM

CSV_PATH = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["download_status"].isin(["ok", "exists"])].head(3).copy()

    model = MiniCPMVLM("openbmb/MiniCPM-V-2_6")

    for _, row in df.iterrows():
        print("\n" + "=" * 100)
        print("QUESTION:", row["question"])
        print("GOLD:", row["short_answer"])
        print("IMAGE:", row["local_image_path"])
        pred = model.answer_image_only(
            image_path=str(row["local_image_path"]),
            question=str(row["question"])
        )
        print("PRED:", pred)


if __name__ == "__main__":
    main()
