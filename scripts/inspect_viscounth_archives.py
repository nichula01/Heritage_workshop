#!/usr/bin/env python3
from pathlib import Path
import zipfile

ROOT = Path("data/raw/viscounth_repo")
TRAIN_DIR = ROOT / "Dataset 2.0" / "English version" / "training set"
DESC_DIR = ROOT / "Desription" / "English Description"

def inspect_zip(path: Path, max_members: int = 20):
    print("\n" + "=" * 100)
    print(f"[ZIP] {path}")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            print(f"members: {len(names)}")
            for name in names[:max_members]:
                print(" ", name)
    except Exception as e:
        print(f"[ERROR] {e}")

def main():
    print("=" * 100)
    print("TRAINING ZIP FILES")
    print("=" * 100)
    train_zips = sorted(TRAIN_DIR.glob("*.zip"))
    for p in train_zips[:8]:
        inspect_zip(p, max_members=15)

    print("\n" + "=" * 100)
    print("DESCRIPTION ZIP FILES")
    print("=" * 100)
    desc_zips = sorted(DESC_DIR.glob("*.zip"))
    for p in desc_zips:
        inspect_zip(p, max_members=15)

    print("\n" + "=" * 100)
    print("RAR FILES (NOT HANDLED YET)")
    print("=" * 100)
    for p in sorted(TRAIN_DIR.glob("*.rar")):
        print(p)
    for p in sorted((ROOT / "Dataset 2.0" / "English version").glob("*.rar")):
        print(p)

if __name__ == "__main__":
    main()
