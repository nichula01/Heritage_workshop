#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.minicpm_vlm import MiniCPMVLM

IN_CSV = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")
OUT_CSV = Path("results/minicpm_v26_image_only/predictions.csv")

MAX_SAMPLES = 50


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df = df[df["download_status"].isin(["ok", "exists"])].head(MAX_SAMPLES).copy()

    model = MiniCPMVLM("openbmb/MiniCPM-V-2_6")

    rows = []
    for i, row in df.iterrows():
        question = str(row["question"])
        image_path = str(row["local_image_path"])

        try:
            pred = model.answer_image_only(image_path=image_path, question=question)
            status = "ok"
        except Exception as e:
            pred = ""
            status = f"error:{type(e).__name__}"

        rows.append({
            "sample_id": row.get("sample_id", ""),
            "template_id": row.get("template_id", ""),
            "question_type": row.get("question_type", ""),
            "question": question,
            "gold_answer": row.get("short_answer", row.get("answer", "")),
            "prediction": pred,
            "image_path": image_path,
            "status": status
        })

        print(f"[{i+1}/{len(df)}] {status}")
        print("Q:", question)
        print("GT:", row.get("short_answer", ""))
        print("PR:", pred)
        print("-" * 80)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] saved predictions: {OUT_CSV}")
    print(out_df.head(10).to_string())


if __name__ == "__main__":
    main()
