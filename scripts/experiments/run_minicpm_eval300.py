#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoTokenizer

INPUT_JSONL = Path("data/processed/viscounth/eval300_aer_input.jsonl")
OUTPUT_JSONL = Path("data/processed/viscounth/eval300_predictions.jsonl")

MODEL_NAME = "openbmb/MiniCPM-V-2_6"

def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

    rows = []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    out_f = open(OUTPUT_JSONL, "w", encoding="utf-8")

    for i, row in enumerate(rows, start=1):

        question = row["question"]
        img_path = row["image_path"]

        try:

            if img_path and Path(img_path).exists():
                image = Image.open(img_path).convert("RGB")
                answer = model.chat(
                    image=image,
                    msgs=[{"role":"user","content":question}],
                    tokenizer=tokenizer
                )
            else:
                answer = model.chat(
                    msgs=[{"role":"user","content":question}],
                    tokenizer=tokenizer
                )

        except Exception as e:
            answer = f"[ERROR] {e}"

        record = {
            **row,
            "model_answer": answer
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{i}/{len(rows)}] done")

    out_f.close()

    print(f"\n[OK] predictions saved: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
