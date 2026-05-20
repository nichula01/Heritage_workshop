#!/usr/bin/env python3
import csv
import json
from pathlib import Path

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
OUT_JSONL = Path("data/processed/viscounth/eval300_aer_input.jsonl")

def main():

    rows = []

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:

        for row in rows:

            record = {
                "sample_id": row.get("sample_id", ""),
                "question": row.get("question", ""),
                "route": row.get("route", row.get("question_type", "")),
                "evidence_mode": row.get("evidence_mode", ""),
                "image_path": row.get("local_image_path", ""),
                "retrieved_text_1": row.get("retrieved_text_1", ""),
                "retrieved_text_2": row.get("retrieved_text_2", "")
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] saved AER input file: {OUT_JSONL}")
    print("samples:", len(rows))


if __name__ == "__main__":
    main()
