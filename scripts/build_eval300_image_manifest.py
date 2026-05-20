#!/usr/bin/env python3
import csv
import hashlib
from pathlib import Path

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300.csv")
OUT_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest.csv")


def make_image_name(image_url: str, depiction_name: str) -> str:
    depiction_name = str(depiction_name or "").strip()
    if depiction_name and depiction_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return depiction_name
    h = hashlib.md5(str(image_url).encode("utf-8")).hexdigest()[:16]
    return f"{h}.jpg"


def main():
    rows = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_url = row.get("image_url", "").strip()
            if not image_url:
                continue

            image_filename = make_image_name(image_url, row.get("depiction_name", ""))
            row["image_filename"] = image_filename
            row["local_image_path"] = f"data/raw/viscounth_images_eval300/{image_filename}"
            row["download_status"] = "pending"
            rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] saved manifest: {OUT_CSV}")
    print(f"[INFO] rows: {len(rows)}")


if __name__ == "__main__":
    main()
