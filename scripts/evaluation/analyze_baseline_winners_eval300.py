#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

IMG_CSV = Path("results/internvl25_2b_image_only_eval300/predictions.csv")
TXT_CSV = Path("results/internvl25_2b_text_only_eval300/predictions.csv")
MM_CSV = Path("results/internvl25_2b_naive_multimodal_eval300/predictions.csv")

OUT_JSON = Path("results/aer_v4/template_policy_eval300.json")

def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def exact_match(gold: str, pred: str) -> int:
    return int(normalize_text(gold) == normalize_text(pred))

def load_scores(path: Path):
    by_template = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tpl = row["template_id"]
            em = exact_match(row.get("gold_answer", ""), row.get("prediction", ""))
            by_template[tpl].append(em)
    return by_template

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def main():
    img = load_scores(IMG_CSV)
    txt = load_scores(TXT_CSV)
    mm = load_scores(MM_CSV)

    all_templates = sorted(set(img) | set(txt) | set(mm))

    policy = {}

    for tpl in all_templates:
        scores = {
            "image_only": mean(img.get(tpl, [])),
            "text_only": mean(txt.get(tpl, [])),
            "image_plus_text": mean(mm.get(tpl, [])),
        }

        best_mode = max(scores, key=scores.get)

        # Conservative text-first policy
        if scores["image_only"] >= scores["text_only"] + 0.10 and scores["image_only"] >= scores["image_plus_text"] + 0.05:
            preferred = "image_only"
        elif scores["image_plus_text"] >= scores["text_only"] + 0.05 and scores["image_plus_text"] >= scores["image_only"] + 0.05:
            preferred = "image_plus_text"
        else:
            preferred = "text_only"

        policy[tpl] = {
            "scores": scores,
            "best_mode_raw": best_mode,
            "preferred_mode_v4": preferred,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    print(f"[OK] saved: {OUT_JSON}")

    for tpl in list(sorted(policy.keys()))[:20]:
        print(tpl, policy[tpl])

if __name__ == "__main__":
    main()
