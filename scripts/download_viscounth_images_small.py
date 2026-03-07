#!/usr/bin/env python3
from pathlib import Path
from io import BytesIO
import time

import pandas as pd
import requests
from PIL import Image

MANIFEST_CSV = Path("data/processed/viscounth/viscounth_en_image_manifest_small.csv")
OUT_DIR = Path("data/raw/viscounth_images_small")
UPDATED_CSV = Path("data/processed/viscounth/viscounth_en_image_manifest_small_downloaded.csv")


def validate_image_bytes(content: bytes) -> bool:
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False


def main():
    df = pd.read_csv(MANIFEST_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    statuses = []
    sizes = []

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 HeritageWorkshop/1.0"})

    for i, row in df.iterrows():
        url = str(row["image_url"]).strip()
        local_path = Path(str(row["local_image_path"]))

        if local_path.exists():
            statuses.append("exists")
            sizes.append(local_path.stat().st_size)
            print(f"[{i+1}/{len(df)}] exists: {local_path}")
            continue

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            content = resp.content

            if not validate_image_bytes(content):
                statuses.append("invalid_image")
                sizes.append(0)
                print(f"[{i+1}/{len(df)}] invalid_image: {url}")
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)

            statuses.append("ok")
            sizes.append(len(content))
            print(f"[{i+1}/{len(df)}] ok: {local_path}")

        except Exception as e:
            statuses.append(f"error:{type(e).__name__}")
            sizes.append(0)
            print(f"[{i+1}/{len(df)}] error: {url} | {e}")

        time.sleep(0.2)

    df["download_status"] = statuses
    df["download_size_bytes"] = sizes
    df["file_exists"] = df["local_image_path"].apply(lambda p: Path(p).exists())

    UPDATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UPDATED_CSV, index=False)

    print(f"\n[OK] saved updated manifest: {UPDATED_CSV}")
    print(df["download_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
