#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.pipeline import AdaptiveEvidenceRouter

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
OUT_CSV = Path("outputs/aer/viscounth_en_eval300_prepared.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare routed AER evidence for the VISCOUNTH English eval subset."
    )
    parser.add_argument("--input-csv", type=Path, default=IN_CSV, help="Downloaded eval manifest CSV.")
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV, help="Prepared evidence CSV to write.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved text sentences per sample.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_csv.exists():
        raise SystemExit(
            f"Input CSV not found: {args.input_csv}. Build the eval subset and image manifest first."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)

    if not rows_in:
        raise SystemExit(f"No downloadable rows found in input CSV: {args.input_csv}")

    aer = AdaptiveEvidenceRouter()
    rows_out = []

    for i, row in enumerate(rows_in, start=1):
        out = aer.prepare(
            sample_id=row["sample_id"],
            template_id=row["template_id"],
            question=row["question"],
            description=row["description"],
            top_k=args.top_k,
        )

        rows_out.append({
            "sample_id": out.sample_id,
            "template_id": out.template_id,
            "question": out.question,
            "route": out.route,
            "route_confidence": out.route_confidence,
            "route_source": out.route_source,
            "route_reason": out.route_reason,
            "evidence_mode": out.evidence_mode,
            "retrieved_text_1": out.retrieved_sentences[0] if len(out.retrieved_sentences) > 0 else "",
            "retrieved_text_2": out.retrieved_sentences[1] if len(out.retrieved_sentences) > 1 else "",
            "retrieval_score_1": out.retrieval_scores[0] if len(out.retrieval_scores) > 0 else "",
            "retrieval_score_2": out.retrieval_scores[1] if len(out.retrieval_scores) > 1 else "",
        })

        if i % 25 == 0:
            print(f"[INFO] prepared {i}/{len(rows_in)}")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] saved prepared evidence: {args.output_csv}")


if __name__ == "__main__":
    main()
