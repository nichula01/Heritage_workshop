#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/viscounth_extracted")
TRAIN_ROOT = ROOT / "english_training"
DESC_ROOT = ROOT / "english_descriptions"


def inspect_csv(path: Path, nrows: int = 5):
    print("\n" + "=" * 100)
    print(path)
    try:
        df = pd.read_csv(path, nrows=nrows)
        print("shape preview:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(nrows).to_string())
    except Exception as e:
        print("[WARN] could not read:", e)


def main():
    train_csvs = sorted(TRAIN_ROOT.rglob("*.csv"))
    desc_csvs = sorted(DESC_ROOT.rglob("*.csv"))

    print("=" * 100)
    print("TRAINING CSV COUNT")
    print("=" * 100)
    print(len(train_csvs))

    print("=" * 100)
    print("DESCRIPTION CSV COUNT")
    print("=" * 100)
    print(len(desc_csvs))

    print("\n" + "=" * 100)
    print("FIRST 8 TRAINING CSV FILES")
    print("=" * 100)
    for p in train_csvs[:8]:
        inspect_csv(p, nrows=3)

    print("\n" + "=" * 100)
    print("DESCRIPTION CSV FILES")
    print("=" * 100)
    for p in desc_csvs:
        inspect_csv(p, nrows=3)


if __name__ == "__main__":
    main()
