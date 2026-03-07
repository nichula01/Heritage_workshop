#!/usr/bin/env python3
from pathlib import Path
import zipfile

ROOT = Path("data/raw/viscounth_repo")
OUT = Path("data/raw/viscounth_extracted")

TRAIN_DIR = ROOT / "Dataset 2.0" / "English version" / "training set"
DESC_DIR = ROOT / "Desription" / "English Description"

TRAIN_OUT = OUT / "english_training"
DESC_OUT = OUT / "english_descriptions"


def extract_zips(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(src_dir.glob("*.zip"))
    print(f"[INFO] extracting {len(zip_files)} zip files from {src_dir}")

    for zp in zip_files:
        target_dir = dst_dir / zp.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(target_dir)
            print(f"[OK] {zp.name} -> {target_dir}")
        except Exception as e:
            print(f"[WARN] failed: {zp.name} | {e}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    extract_zips(TRAIN_DIR, TRAIN_OUT)
    extract_zips(DESC_DIR, DESC_OUT)
    print("\n[DONE] extraction finished")
    print(TRAIN_OUT)
    print(DESC_OUT)


if __name__ == "__main__":
    main()
