#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.answer_extractor import extract_answer
from src.aer.internvl_vlm import InternVLVLM
from src.aer.prediction_normalizer import normalize_prediction

MANIFEST_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
PREPARED_CSV = Path("outputs/aer/viscounth_en_eval300_prepared.csv")
POLICY_JSON = Path("results/aer_v4/template_policy_eval300.json")

OUT_CSV = Path("results/internvl25_2b_aer_v4_1_eval300/predictions.csv")

def load_prepared_map(path: Path):

    prepared = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prepared[row["sample_id"]] = row

    return prepared

def load_policy(path: Path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    prepared_map = load_prepared_map(PREPARED_CSV)
    policy = load_policy(POLICY_JSON)

    rows_in = []

    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)

    print(f"[INFO] usable rows: {len(rows_in)}")

    model = InternVLVLM("OpenGVLab/InternVL2_5-2B")

    rows_out = []

    for i, row in enumerate(rows_in, start=1):

        sample_id = row["sample_id"]
        template_id = row["template_id"]
        question = row["question"]
        image_path = row["local_image_path"]
        full_description = row["description"]

        prep = prepared_map.get(sample_id, {})

        retrieved_1 = (prep.get("retrieved_text_1", "") or "").strip()
        retrieved_2 = (prep.get("retrieved_text_2", "") or "").strip()

        retrieved_text = " ".join([x for x in [retrieved_1, retrieved_2] if x]).strip()

        tpl_policy = policy.get(template_id, {})
        preferred_mode = tpl_policy.get("preferred_mode_v4", "text_only")

        if template_id == "SHAPE":
            preferred_mode = "image_only"

        pred = ""
        answer_source = ""
        status = "ok"

        try:

            if retrieved_text:

                pred = extract_answer(
                    question=question,
                    evidence_text=retrieved_text,
                    template_id=template_id
                ) or ""

                if pred:
                    answer_source = "extractor_retrieved"

            if not pred:

                pred = extract_answer(
                    question=question,
                    evidence_text=full_description,
                    template_id=template_id
                ) or ""

                if pred:
                    answer_source = "extractor_full_description"

            if not pred:

                if preferred_mode == "image_only":

                    pred = model.answer_image_only(
                        image_path=image_path,
                        question=question
                    )

                    answer_source = "vlm_image"

                elif preferred_mode == "image_plus_text":

                    pred = model.answer_image_plus_text(
                        image_path=image_path,
                        question=question,
                        evidence_text=retrieved_text if retrieved_text else full_description
                    )

                    answer_source = "vlm_multimodal"

                else:

                    pred = model.answer_text_only(
                        question=question,
                        evidence_text=full_description
                    )

                    answer_source = "vlm_text"

            pred = normalize_prediction(pred)

        except Exception as e:

            pred = ""
            answer_source = "error"
            status = f"error:{type(e).__name__}"

        rows_out.append({
            "sample_id": sample_id,
            "template_id": template_id,
            "question_type": row.get("question_type", ""),
            "question": question,
            "gold_answer": row.get("short_answer", row.get("answer", "")),
            "prediction": pred,
            "preferred_mode": preferred_mode,
            "answer_source": answer_source,
            "status": status,
        })

        if i % 25 == 0 or i == len(rows_in):
            print(f"[INFO] aer-v4.1 {i}/{len(rows_in)}")

    fieldnames = [
        "sample_id",
        "template_id",
        "question_type",
        "question",
        "gold_answer",
        "prediction",
        "preferred_mode",
        "answer_source",
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
