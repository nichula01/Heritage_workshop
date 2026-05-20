#!/usr/bin/env python3
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

IN_CSV = Path("data/processed/viscounth/viscounth_en_merged.csv")
OUT_CSV = Path("data/processed/viscounth/viscounth_en_eval300.csv")
STATS_JSON = Path("data/processed/viscounth/viscounth_en_eval300_stats.json")

TARGETS = {
    "visual": 100,
    "contextual": 100,
    "mixed": 100,
}

SEED = 42


def main():
    random.seed(SEED)

    rows = []

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_url = row.get("image_url", "").strip()

            if not image_url:
                continue

            rows.append(row)

    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        qtype = row.get("question_type", "mixed")
        template_id = row.get("template_id", "UNKNOWN")

        grouped[qtype][template_id].append(row)

    for qtype in grouped:
        for template_id in grouped[qtype]:
            random.shuffle(grouped[qtype][template_id])

    selected = []
    used_images = set()
    used_assets = set()

    for qtype, target_n in TARGETS.items():

        template_ids = sorted(grouped[qtype].keys())

        picked = 0

        while picked < target_n:

            made_progress = False

            for template_id in template_ids:

                bucket = grouped[qtype][template_id]

                while bucket:

                    row = bucket.pop(0)

                    image_url = row.get("image_url", "").strip()
                    asset = row.get("cultural_property", "").strip()

                    if image_url in used_images:
                        continue

                    if asset in used_assets:
                        continue

                    selected.append(row)

                    used_images.add(image_url)
                    used_assets.add(asset)

                    picked += 1
                    made_progress = True

                    break

                if picked >= target_n:
                    break

            if not made_progress:
                print(f"[WARN] Could not fully reach target for {qtype}. Picked {picked}/{target_n}")
                break

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    qtype_counts = Counter(r.get("question_type", "") for r in selected)
    template_counts = Counter(r.get("template_id", "") for r in selected)

    stats = {
        "num_rows": len(selected),
        "question_type_counts": dict(sorted(qtype_counts.items())),
        "num_unique_images": len({r.get("image_url", "") for r in selected}),
        "num_unique_assets": len({r.get("cultural_property", "") for r in selected}),
        "num_unique_templates": len(template_counts),
        "top_templates": template_counts.most_common(20),
    }

    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved: {OUT_CSV}")
    print(f"[OK] saved: {STATS_JSON}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
