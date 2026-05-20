#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.answer_extractor import extract_answer
from src.aer.internvl_vlm import InternVLVLM

MANIFEST_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
PREPARED_CSV = Path("outputs/aer/viscounth_en_eval300_prepared.csv")
OUT_CSV = Path("results/internvl25_2b_routed_aer_v3_eval300/predictions.csv")


def load_prepared_map(path: Path):
    prepared = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prepared[row["sample_id"]] = row
    return prepared


def main():

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    prepared_map = load_prepared_map(PREPARED_CSV)

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
        question = row["question"]
        image_path = row["local_image_path"]
        full_description = row["description"]

        prep = prepared_map.get(sample_id, {})

        route = prep.get("route", row.get("question_type", "mixed"))
        evidence_mode = prep.get("evidence_mode", "image_plus_text")

        retrieved_1 = (prep.get("retrieved_text_1", "") or "").strip()
        retrieved_2 = (prep.get("retrieved_text_2", "") or "").strip()

        evidence_text = " ".join([x for x in [retrieved_1, retrieved_2] if x])

        if not evidence_text.strip():
            evidence_text = full_description

        pred = ""
        answer_source = ""
        status = "ok"

        try:

            if route == "visual":

                pred = model.answer_image_only(
                    image_path=image_path,
                    question=question
                )

                answer_source = "vlm_image"

            elif route == "contextual":

                pred = model.answer_text_only(
                    question=question,
                    evidence_text=evidence_text
                )

                answer_source = "vlm_text"

                extracted = extract_answer(
                    question=question,
                    evidence_text=evidence_text,
                    route=route
                )

                if (not pred.strip() or pred.strip().lower() in {
                    "unknown",
                    "not specified",
                    "not provided",
                    "i do not know"
                }) and extracted:

                    pred = extracted
                    answer_source = "extractor_fallback"

            else:

                extracted = extract_answer(
                    question=question,
                    evidence_text=evidence_text,
                    route=route
                )

                if extracted:

                    pred = extracted
                    answer_source = "extractor"

                else:

                    pred = model.answer_image_plus_text(
                        image_path=image_path,
                        question=question,
                        evidence_text=evidence_text
                    )

                    answer_source = "vlm_multimodal"

        except Exception as e:

            pred = ""
            answer_source = "error"
            status = f"error:{type(e).__name__}"

        rows_out.append({
            "sample_id": sample_id,
            "template_id": row.get("template_id", ""),
            "question_type": row.get("question_type", ""),
            "route": route,
            "evidence_mode": evidence_mode,
            "question": question,
            "gold_answer": row.get("short_answer", row.get("answer", "")),
            "prediction": pred,
            "image_path": image_path,
            "evidence_text": evidence_text,
            "answer_source": answer_source,
            "status": status,
        })

        if i % 25 == 0 or i == len(rows_in):
            print(f"[INFO] routed aerv3 {i}/{len(rows_in)}")

    fieldnames = [
        "sample_id",
        "template_id",
        "question_type",
        "route",
        "evidence_mode",
        "question",
        "gold_answer",
        "prediction",
        "image_path",
        "evidence_text",
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
