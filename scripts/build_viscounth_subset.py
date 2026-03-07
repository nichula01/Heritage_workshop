#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


TRAIN_ROOT = Path("data/raw/viscounth_extracted/english_training")
DESC_ROOT = Path("data/raw/viscounth_extracted/english_descriptions")
ROUTE_MAP_PATH = Path("metadata/viscounth_route_map.json")
OUT_DIR = Path("data/processed/viscounth")


def normalize_text(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def load_route_map():
    with open(ROUTE_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training():
    frames = []
    csv_files = sorted(TRAIN_ROOT.rglob("*.csv"))
    print(f"[INFO] training csv files: {len(csv_files)}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["template_id"] = df["Id"].astype(str).str.strip()
        df["source_csv"] = str(csv_path)
        df["row_in_source"] = range(len(df))
        frames.append(df)

    train_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] total training rows: {len(train_df)}")
    return train_df


def load_descriptions():
    frames = []
    csv_files = sorted(DESC_ROOT.rglob("*.csv"))
    print(f"[INFO] description csv files: {len(csv_files)}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["desc_source_csv"] = str(csv_path)
        frames.append(df)

    desc_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] total description rows: {len(desc_df)}")
    return desc_df


def build_merged(max_per_template=100):
    route_map = load_route_map()
    train_df = load_training()
    desc_df = load_descriptions()

    # normalize join keys
    for col in ["CulturalProperty", "Depiction"]:
        train_df[col] = train_df[col].astype(str).str.strip()
        desc_df[col] = desc_df[col].astype(str).str.strip()

    # exact merge on property + depiction
    desc_exact = desc_df.rename(
        columns={
            "Description": "description",
            "Depiction_Name": "depiction_name",
            "desc_source_csv": "description_source_csv"
        }
    )

    merged = train_df.merge(
        desc_exact[
            ["CulturalProperty", "Depiction", "description", "depiction_name", "description_source_csv"]
        ],
        on=["CulturalProperty", "Depiction"],
        how="left"
    )

    # fallback merge on CulturalProperty only for rows still missing description
    prop_only = (
        desc_exact.sort_values("CulturalProperty")
        .drop_duplicates(subset=["CulturalProperty"])
        [["CulturalProperty", "description", "depiction_name", "description_source_csv"]]
        .rename(
            columns={
                "description": "description_fallback",
                "depiction_name": "depiction_name_fallback",
                "description_source_csv": "description_source_csv_fallback"
            }
        )
    )

    merged = merged.merge(prop_only, on="CulturalProperty", how="left")

    merged["description"] = merged["description"].fillna(merged["description_fallback"])
    merged["depiction_name"] = merged["depiction_name"].fillna(merged["depiction_name_fallback"])
    merged["description_source_csv"] = merged["description_source_csv"].fillna(
        merged["description_source_csv_fallback"]
    )

    merged["question"] = merged["Question"].map(normalize_text)
    merged["answer"] = merged["Answer"].map(normalize_text)
    merged["short_answer"] = merged["Short Answer"].map(normalize_text)
    merged["description"] = merged["description"].map(normalize_text)

    merged["question_type"] = merged["template_id"].map(route_map).fillna("mixed")
    merged["question_type_source"] = merged["template_id"].apply(
        lambda x: "template_map_v1" if x in route_map else "default_mixed"
    )

    merged["sample_id"] = [
        f"{tpl}_{i:08d}" for i, tpl in enumerate(merged["template_id"].tolist())
    ]

    keep_cols = [
        "sample_id",
        "template_id",
        "question_type",
        "question_type_source",
        "question",
        "answer",
        "short_answer",
        "Depiction",
        "depiction_name",
        "CulturalProperty",
        "CulturalProperty Class",
        "Typology",
        "description",
        "Short Answer_index",
        "source_csv",
        "description_source_csv"
    ]

    merged = merged[keep_cols].copy()
    merged = merged.rename(
        columns={
            "Depiction": "image_url",
            "CulturalProperty": "cultural_property",
            "CulturalProperty Class": "cultural_property_class",
            "Typology": "typology",
            "Short Answer_index": "short_answer_index"
        }
    )

    merged["has_description"] = merged["description"].fillna("").str.len() > 0
    merged["has_image_url"] = merged["image_url"].fillna("").str.len() > 0

    debug_subset = (
        merged.groupby("template_id", group_keys=False)
        .head(max_per_template)
        .reset_index(drop=True)
    )

    return merged, debug_subset


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged, debug_subset = build_merged(max_per_template=50)

    full_csv = OUT_DIR / "viscounth_en_merged.csv"
    debug_csv = OUT_DIR / "viscounth_en_debug50_per_template.csv"
    stats_json = OUT_DIR / "viscounth_en_stats.json"

    merged.to_csv(full_csv, index=False)
    debug_subset.to_csv(debug_csv, index=False)

    stats = {
        "num_rows_full": int(len(merged)),
        "num_rows_debug": int(len(debug_subset)),
        "num_templates": int(merged["template_id"].nunique()),
        "num_rows_with_description": int(merged["has_description"].sum()),
        "num_rows_with_image_url": int(merged["has_image_url"].sum()),
        "question_type_counts": merged["question_type"].value_counts().to_dict(),
        "template_counts": merged["template_id"].value_counts().to_dict()
    }

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved full merged csv: {full_csv}")
    print(f"[OK] saved debug subset csv: {debug_csv}")
    print(f"[OK] saved stats json: {stats_json}")
    print("\n[INFO] head of merged data:")
    print(merged.head(5).to_string())


if __name__ == "__main__":
    main()
