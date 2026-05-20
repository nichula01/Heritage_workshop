#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

def norm_text(s: str) -> str:
    s = str(s or "").strip().lower()
    return " ".join(s.split())

def em_score(gold: str, pred: str) -> float:
    return 1.0 if norm_text(gold) == norm_text(pred) else 0.0

def contains_score(gold: str, pred: str) -> float:
    g = norm_text(gold)
    p = norm_text(pred)
    if not g or not p:
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0

def load_predictions(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def stratified_sample(rows, n, rng):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)

    types = sorted(by_type.keys())
    total = len(rows)

    alloc = {}
    remainders = []
    assigned = 0

    for t in types:
        raw = n * len(by_type[t]) / total
        base = int(math.floor(raw))
        alloc[t] = base
        assigned += base
        remainders.append((raw - base, t))

    remainders.sort(reverse=True)
    left = n - assigned

    for _, t in remainders[:left]:
        alloc[t] += 1

    sample = []

    for t in types:
        k = min(alloc[t], len(by_type[t]))
        sample.extend(rng.sample(by_type[t], k))

    if len(sample) < n:
        used_ids = set(id(x) for x in sample)
        leftovers = [r for r in rows if id(r) not in used_ids]
        need = n - len(sample)
        sample.extend(rng.sample(leftovers, need))

    return sample

def evaluate_rows(rows):
    ems = [em_score(r["gold_answer"], r["prediction"]) for r in rows]
    cons = [contains_score(r["gold_answer"], r["prediction"]) for r in rows]

    return {
        "em": 100.0 * sum(ems) / len(ems),
        "contains": 100.0 * sum(cons) / len(cons),
        "n": len(rows),
    }

def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, math.sqrt(var)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="50,100,150,200,250,298")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-json", type=str, default="results/sample_size_robustness/robustness.json")
    parser.add_argument("--out-csv", type=str, default="results/sample_size_robustness/robustness.csv")

    args = parser.parse_args()

    method_files = {
        "Image-only": Path("results/internvl25_2b_image_only_eval300/predictions.csv"),
        "Text-only": Path("results/internvl25_2b_text_only_eval300/predictions.csv"),
        "Naive multimodal": Path("results/internvl25_2b_naive_multimodal_eval300/predictions.csv"),
        "AER v4.1": Path("results/aer_v4_1_ref/predictions.csv"),
        "AER full-only": Path("results/ab_extractor_full_only/predictions.csv"),
    }

    loaded = {name: load_predictions(path) for name, path in method_files.items()}

    ref_name = list(loaded.keys())[0]
    ref_ids = [r["sample_id"] for r in loaded[ref_name]]

    for name, rows in loaded.items():
        ids = [r["sample_id"] for r in rows]
        if ids != ref_ids:
            raise ValueError(f"Sample order mismatch for {name}")

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    out = {}
    rng_master = random.Random(args.seed)

    for size in sizes:
        out[size] = {}

        seeds = [rng_master.randint(0, 10**9) for _ in range(args.trials)]

        for method_name, rows in loaded.items():
            em_vals = []
            contains_vals = []

            for s in seeds:
                rng = random.Random(s)
                subset = stratified_sample(rows, size, rng)
                res = evaluate_rows(subset)

                em_vals.append(res["em"])
                contains_vals.append(res["contains"])

            em_mean, em_std = mean_std(em_vals)
            c_mean, c_std = mean_std(contains_vals)

            out[size][method_name] = {
                "em_mean": em_mean,
                "em_std": em_std,
                "contains_mean": c_mean,
                "contains_std": c_std,
                "trials": args.trials,
            }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["size", "method", "em_mean", "em_std", "contains_mean", "contains_std", "trials"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for method_name, stats in out[size].items():
                writer.writerow({
                    "size": size,
                    "method": method_name,
                    **stats
                })

    print(f"[OK] saved json: {out_json}")
    print(f"[OK] saved csv: {out_csv}")

    print("\n=== EM mean +/- std by sample size ===")

    for size in sizes:
        print(f"\n[size={size}]")
        for method_name, stats in out[size].items():
            print(
                f"{method_name:18s} "
                f"EM={stats['em_mean']:.2f}±{stats['em_std']:.2f} "
                f"Contains={stats['contains_mean']:.2f}±{stats['contains_std']:.2f}"
            )

if __name__ == "__main__":
    main()
