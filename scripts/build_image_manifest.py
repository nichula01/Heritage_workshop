#!/usr/bin/env python3
import hashlib
from pathlib import Path

import pandas as pd

IN_CSV = Path("data/processed/viscounth/viscounth_en_debug50_per_template.csv")
OUT_CSV = Path("data/processed/viscounth/viscounth_en_image_manifest_small.csv")

PER_TEMPLATE = 5


def make_image_name(image_url: str, depiction_name: str) -> str:
    depiction_name = str(depiction_name or "").strip()
    if depiction_name and depiction_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return depiction_name

    h = hashlib.md5(str(image_url).encode("utf-8")).hexdigest()[:16]
    return f"{h}.jpg"


def main():
    df = pd.read_csv(IN_CSV)

    # keep a small subset per template for initial download / debugging
    df = (
        df.groupby("template_id", group_keys=False)
        .head(PER_TEMPLATE)
        .reset_index(drop=True)
    )

    # deduplicate by image URL so we don't download same image repeatedly
    df = df.drop_duplicates(subset=["image_url"]).reset_index(drop=True)

    df["image_filename"] = [
        make_image_name(url, dep_name)
        for url, dep_name in zip(df["image_url"], df.get("depiction_name", [""] * len(df)))
    ]
    df["local_image_path"] = df["image_filename"].apply(
        lambda x: f"data/raw/viscounth_images_small/{x}"
    )
    df["download_status"] = "pending"

    keep_cols = [
        "sample_id",
        "template_id",
        "question_type",
        "question",
        "answer",
        "short_answer",
        "image_url",
        "depiction_name",
        "cultural_property",
        "description",
        "image_filename",
        "local_image_path",
        "download_status",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[OK] saved manifest: {OUT_CSV}")
    print(f"[INFO] rows: {len(out_df)}")
    print(out_df.head(10).to_string())


if __name__ == "__main__":
    main()
