#!/usr/bin/env python3
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def exact_match(gold: str, pred: str) -> int:
    return int(normalize_text(gold) == normalize_text(pred))


def contains_match(gold: str, pred: str) -> int:
    g = normalize_text(gold)
    p = normalize_text(pred)
    if not g or not p:
        return 0
    return int(g in p or p in g)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/evaluate_predictions_basic.py <predictions_csv>")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold = row.get("gold_answer", "")
            pred = row.get("prediction", "")
            row["em"] = exact_match(gold, pred)
            row["contains"] = contains_match(gold, pred)
            rows.append(row)

    print("rows:", len(rows))
    print("EM:", round(mean([r["em"] for r in rows]), 4))
    print("Contains:", round(mean([r["contains"] for r in rows]), 4))

    by_qtype = defaultdict(list)
    by_template = defaultdict(list)

    for r in rows:
        by_qtype[r.get("question_type", "unknown")].append(r)
        by_template[r.get("template_id", "unknown")].append(r)

    print("\nBy question_type")
    for k in sorted(by_qtype):
        em = mean([r["em"] for r in by_qtype[k]])
        cm = mean([r["contains"] for r in by_qtype[k]])
        print(f"{k:15s} EM={em:.4f}  Contains={cm:.4f}  n={len(by_qtype[k])}")

    print("\nBy template_id")
    for k in sorted(by_template):
        em = mean([r["em"] for r in by_template[k]])
        cm = mean([r["contains"] for r in by_template[k]])
        print(f"{k:20s} EM={em:.4f}  Contains={cm:.4f}  n={len(by_template[k])}")


if __name__ == "__main__":
    main()
