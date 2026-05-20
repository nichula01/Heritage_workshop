#!/usr/bin/env python3
import csv
import time
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image

IN_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest.csv")
OUT_CSV = Path("data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv")
OUT_DIR = Path("data/raw/viscounth_images_eval300")

def validate_image_bytes(content: bytes) -> bool:
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    for i, row in enumerate(rows, start=1):

        url = row["image_url"].strip()
        local_path = Path(row["local_image_path"])

        if local_path.exists():
            row["download_status"] = "exists"
            print(f"[{i}/{len(rows)}] exists: {local_path}")
            continue

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                content = resp.read()

            if not validate_image_bytes(content):
                row["download_status"] = "invalid_image"
                print(f"[{i}/{len(rows)}] invalid_image: {url}")
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(content)

            row["download_status"] = "ok"
            print(f"[{i}/{len(rows)}] ok: {local_path}")

        except Exception as e:
            row["download_status"] = f"error:{type(e).__name__}"
            print(f"[{i}/{len(rows)}] error: {url} | {e}")

        time.sleep(0.15)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] saved updated manifest: {OUT_CSV}")

if __name__ == "__main__":
    main()
