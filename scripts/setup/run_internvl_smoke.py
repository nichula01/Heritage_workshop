#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.internvl_vlm import InternVLVLM

CSV_PATH = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")


def main():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows.append(row)
            if len(rows) == 3:
                break

    model = InternVLVLM("OpenGVLab/InternVL2_5-2B")

    for row in rows:
        print("\n" + "=" * 100)
        print("QUESTION:", row["question"])
        print("GOLD:", row["short_answer"])
        print("IMAGE:", row["local_image_path"])
        pred = model.answer_image_only(
            image_path=row["local_image_path"],
            question=row["question"]
        )
        print("PRED:", pred)


if __name__ == "__main__":
    main()
