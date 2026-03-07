#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aer.pipeline import AdaptiveEvidenceRouter


IN_CSV = Path("data/processed/viscounth/viscounth_en_debug50_per_template.csv")
OUT_CSV = Path("outputs/aer/viscounth_en_debug50_prepared.csv")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    aer = AdaptiveEvidenceRouter()

    rows = []
    for _, row in df.iterrows():
        out = aer.prepare(
            sample_id=str(row["sample_id"]),
            template_id=str(row["template_id"]),
            question=str(row["question"]),
            description=str(row["description"]),
            top_k=2
        )

        rows.append({
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
            "retrieval_score_2": out.retrieval_scores[1] if len(out.retrieval_scores) > 1 else ""
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[OK] saved prepared evidence file: {OUT_CSV}")
    print(out_df.head(10).to_string())


if __name__ == "__main__":
    main()
