#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.pipeline import AdaptiveEvidenceRouter

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
OUT_CSV = Path("outputs/aer/viscounth_en_eval300_prepared.csv")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("download_status") in {"ok", "exists"}:
                rows_in.append(row)

    aer = AdaptiveEvidenceRouter()
    rows_out = []

    for i, row in enumerate(rows_in, start=1):
        out = aer.prepare(
            sample_id=row["sample_id"],
            template_id=row["template_id"],
            question=row["question"],
            description=row["description"],
            top_k=2,
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

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] saved prepared evidence: {OUT_CSV}")


if __name__ == "__main__":
    main()
