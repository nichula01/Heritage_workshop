#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

IN_CSV = Path("results/sample_size_robustness/robustness.csv")
OUT_PNG = Path("results/sample_size_robustness/sample_size_em_curve.png")
OUT_PDF = Path("results/sample_size_robustness/sample_size_em_curve.pdf")

def main():
    data = defaultdict(list)

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]

            data[method].append({
                "size": int(row["size"]),
                "em_mean": float(row["em_mean"]),
                "em_std": float(row["em_std"]),
            })

    plt.figure(figsize=(7.2, 4.6))

    order = [
        "Image-only",
        "Text-only",
        "Naive multimodal",
        "AER ",
        "AER-TextFirst",
    ]

    for method in order:
        rows = sorted(data[method], key=lambda x: x["size"])

        xs = [r["size"] for r in rows]
        ys = [r["em_mean"] for r in rows]
        yerr = [r["em_std"] for r in rows]

        plt.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            markersize=4,
            capsize=3,
            label=method
        )

    plt.xlabel("Evaluation subset size")
    plt.ylabel("Exact Match (%)")
    plt.title("Robustness to evaluation subset size")
    plt.legend(frameon=False)

    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

    print(f"[OK] saved: {OUT_PNG}")
    print(f"[OK] saved: {OUT_PDF}")

if __name__ == "__main__":
    main()
