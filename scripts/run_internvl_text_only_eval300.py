#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.internvl_vlm import InternVLVLM

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
OUT_CSV = Path("results/internvl25_2b_text_only_eval300/predictions.csv")

def main():

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)

    print(f"[INFO] usable rows: {len(rows_in)}")

    model = InternVLVLM("OpenGVLab/InternVL2_5-2B")

    rows_out = []

    for i, row in enumerate(rows_in, start=1):
        try:
            pred = model.answer_text_only(
                question=row["question"],
                evidence_text=row["description"]
            )
            status = "ok"

        except Exception as e:
            pred = ""
            status = f"error:{type(e).__name__}"

        rows_out.append({
            "sample_id": row["sample_id"],
            "template_id": row["template_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row.get("short_answer", row.get("answer", "")),
            "prediction": pred,
            "status": status,
        })

        if i % 25 == 0 or i == len(rows_in):
            print(f"[INFO] text-only {i}/{len(rows_in)}")

    fieldnames = [
        "sample_id",
        "template_id",
        "question_type",
        "question",
        "gold_answer",
        "prediction",
        "status",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] saved predictions: {OUT_CSV}")
    print(f"[OK] rows written: {len(rows_out)}")

if __name__ == "__main__":
    main()
