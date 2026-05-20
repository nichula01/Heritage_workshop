#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.internvl_vlm import InternVLVLM

IN_CSV = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")
OUT_CSV = Path("results/internvl25_2b_naive_multimodal/predictions.csv")

MAX_SAMPLES = 50


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)
            if len(rows_in) >= MAX_SAMPLES:
                break

    model = InternVLVLM("OpenGVLab/InternVL2_5-2B")

    fieldnames = [
        "sample_id",
        "template_id",
        "question_type",
        "question",
        "gold_answer",
        "prediction",
        "image_path",
        "status",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows_in, start=1):
            question = row["question"]
            description = row["description"]
            image_path = row["local_image_path"]

            try:
                pred = model.answer_image_plus_text(
                    image_path=image_path,
                    question=question,
                    evidence_text=description
                )
                status = "ok"
            except Exception as e:
                pred = ""
                status = f"error:{type(e).__name__}"

            out_row = {
                "sample_id": row.get("sample_id", ""),
                "template_id": row.get("template_id", ""),
                "question_type": row.get("question_type", ""),
                "question": question,
                "gold_answer": row.get("short_answer", row.get("answer", "")),
                "prediction": pred,
                "image_path": image_path,
                "status": status,
            }
            writer.writerow(out_row)

            print(f"[{i}/{len(rows_in)}] {status}")
            print("Q:", question)
            print("GT:", row.get("short_answer", ""))
            print("PR:", pred)
            print("-" * 80)

    print(f"\n[OK] saved predictions: {OUT_CSV}")


if __name__ == "__main__":
    main()
