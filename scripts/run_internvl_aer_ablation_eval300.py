#!/usr/bin/env python3
import argparse
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


def route_to_mode(route: str) -> str:
    route = str(route or "").strip().lower()
    if route == "visual":
        return "image_only"
    if route == "contextual":
        return "text_only"
    return "image_plus_text"


def choose_mode(template_id, question_type, prepared_row, policy, policy_mode, shape_override):
    # Optional manual override used in AER v4.1
    if shape_override == "on" and template_id == "SHAPE":
        return "image_only", "shape_override"

    if policy_mode == "template":
        mode = policy.get(template_id, {}).get("preferred_mode_v4", "text_only")
        return mode, "template_policy"

    if policy_mode == "textonly":
        return "text_only", "global_textonly"

    if policy_mode == "route":
        route = prepared_row.get("route", question_type)
        return route_to_mode(route), "route_policy"

    raise ValueError(f"Unknown policy_mode: {policy_mode}")


def get_retrieved_text(prepared_row):
    r1 = (prepared_row.get("retrieved_text_1", "") or "").strip()
    r2 = (prepared_row.get("retrieved_text_2", "") or "").strip()
    return " ".join([x for x in [r1, r2] if x]).strip()


def run_extractor(question, template_id, full_description, retrieved_text, extractor_mode):
    if extractor_mode == "off":
        return "", ""

    if extractor_mode == "retrieved_only":
        if retrieved_text:
            pred = extract_answer(
                question=question,
                evidence_text=retrieved_text,
                template_id=template_id,
            ) or ""
            if pred:
                return pred, "extractor_retrieved_only"
        return "", ""

    if extractor_mode == "full_only":
        pred = extract_answer(
            question=question,
            evidence_text=full_description,
            template_id=template_id,
        ) or ""
        if pred:
            return pred, "extractor_full_only"
        return "", ""

    if extractor_mode == "two_stage":
        if retrieved_text:
            pred = extract_answer(
                question=question,
                evidence_text=retrieved_text,
                template_id=template_id,
            ) or ""
            if pred:
                return pred, "extractor_retrieved"
        pred = extract_answer(
            question=question,
            evidence_text=full_description,
            template_id=template_id,
        ) or ""
        if pred:
            return pred, "extractor_full_description"
        return "", ""

    raise ValueError(f"Unknown extractor_mode: {extractor_mode}")


def maybe_normalize(pred, normalize_flag):
    if normalize_flag == "on":
        return normalize_prediction(pred)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True,
                        help="Result folder tag under results/")
    parser.add_argument("--policy-mode", type=str, default="template",
                        choices=["template", "textonly", "route"],
                        help="template=use learned template policy; textonly=always fallback to text; route=original visual/contextual/mixed mapping")
    parser.add_argument("--extractor-mode", type=str, default="two_stage",
                        choices=["off", "retrieved_only", "full_only", "two_stage"],
                        help="How the template-aware extractor is used")
    parser.add_argument("--multimodal-text", type=str, default="top2",
                        choices=["top2", "full"],
                        help="Text source for image+text fallback")
    parser.add_argument("--normalize", type=str, default="on",
                        choices=["on", "off"],
                        help="Apply final answer normalization or not")
    parser.add_argument("--shape-override", type=str, default="on",
                        choices=["on", "off"],
                        help="Whether to force SHAPE -> image_only")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap for debugging; 0 means all usable rows")
    args = parser.parse_args()

    out_csv = Path(f"results/{args.tag}/predictions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    prepared_map = load_prepared_map(PREPARED_CSV)
    policy = load_policy(POLICY_JSON)

    rows_in = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)

    if args.max_samples > 0:
        rows_in = rows_in[:args.max_samples]

    print("[INFO] configuration:")
    print(json.dumps({
        "tag": args.tag,
        "policy_mode": args.policy_mode,
        "extractor_mode": args.extractor_mode,
        "multimodal_text": args.multimodal_text,
        "normalize": args.normalize,
        "shape_override": args.shape_override,
        "usable_rows": len(rows_in),
    }, indent=2))

    model = InternVLVLM("OpenGVLab/InternVL2_5-2B")
    rows_out = []

    for i, row in enumerate(rows_in, start=1):
        sample_id = row["sample_id"]
        template_id = row["template_id"]
        question_type = row.get("question_type", "")
        question = row["question"]
        image_path = row["local_image_path"]
        full_description = row["description"]

        prepared_row = prepared_map.get(sample_id, {})
        retrieved_text = get_retrieved_text(prepared_row)

        preferred_mode, mode_source = choose_mode(
            template_id=template_id,
            question_type=question_type,
            prepared_row=prepared_row,
            policy=policy,
            policy_mode=args.policy_mode,
            shape_override=args.shape_override,
        )

        pred = ""
        answer_source = ""
        status = "ok"

        try:
            # 1) extractor stage
            pred, answer_source = run_extractor(
                question=question,
                template_id=template_id,
                full_description=full_description,
                retrieved_text=retrieved_text,
                extractor_mode=args.extractor_mode,
            )

            # 2) fallback stage
            if not pred:
                if preferred_mode == "image_only":
                    pred = model.answer_image_only(
                        image_path=image_path,
                        question=question,
                    )
                    answer_source = "vlm_image"

                elif preferred_mode == "image_plus_text":
                    evidence_text = retrieved_text if (args.multimodal_text == "top2" and retrieved_text) else full_description
                    pred = model.answer_image_plus_text(
                        image_path=image_path,
                        question=question,
                        evidence_text=evidence_text,
                    )
                    answer_source = f"vlm_multimodal_{args.multimodal_text}"

                else:
                    pred = model.answer_text_only(
                        question=question,
                        evidence_text=full_description,
                    )
                    answer_source = "vlm_text"

            pred = maybe_normalize(pred, args.normalize)

        except Exception as e:
            pred = ""
            answer_source = "error"
            status = f"error:{type(e).__name__}"

        rows_out.append({
            "sample_id": sample_id,
            "template_id": template_id,
            "question_type": question_type,
            "question": question,
            "gold_answer": row.get("short_answer", row.get("answer", "")),
            "prediction": pred,
            "preferred_mode": preferred_mode,
            "mode_source": mode_source,
            "answer_source": answer_source,
            "policy_mode": args.policy_mode,
            "extractor_mode": args.extractor_mode,
            "multimodal_text": args.multimodal_text,
            "normalize_flag": args.normalize,
            "shape_override": args.shape_override,
            "status": status,
        })

        if i % 25 == 0 or i == len(rows_in):
            print(f"[INFO] {args.tag} {i}/{len(rows_in)}")

    fieldnames = [
        "sample_id",
        "template_id",
        "question_type",
        "question",
        "gold_answer",
        "prediction",
        "preferred_mode",
        "mode_source",
        "answer_source",
        "policy_mode",
        "extractor_mode",
        "multimodal_text",
        "normalize_flag",
        "shape_override",
        "status",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] saved predictions: {out_csv}")
    print(f"[OK] rows written: {len(rows_out)}")


if __name__ == "__main__":
    main()
