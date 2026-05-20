# Dataset Setup

VISCOUNTH files are not included in this repository. Download the dataset separately from its official source and follow its license and usage terms.

## Expected Local Layout

Place or extract local files under:

```text
data/
├── raw/
│   ├── viscounth_repo/
│   │   ├── Dataset 2.0/
│   │   └── Desription/
│   ├── viscounth_extracted/
│   │   ├── english_training/
│   │   └── english_descriptions/
│   ├── viscounth_images_small/
│   └── viscounth_images_eval300/
└── processed/
    └── viscounth/
```

The spelling `Desription` reflects the folder name expected by the current VISCOUNTH extraction scripts. If your local copy differs, adjust the local folder name or script constants.

## Git Ignore Policy

Git intentionally ignores:

- `data/raw/`
- `data/processed/`
- archives and extracted dataset files
- downloaded images
- generated CSV/JSONL files

Only `data/README.md` should be tracked.

## Build Workflow

Inspect the local archive layout:

```bash
bash scripts/data/check_viscounth_layout.sh
python scripts/data/inspect_viscounth_archives.py
```

Extract English training and description files:

```bash
python scripts/data/extract_viscounth_english.py
```

Inspect extracted CSVs:

```bash
python scripts/data/inspect_viscounth_csvs.py
```

Build the merged English subset:

```bash
python scripts/data/build_viscounth_subset.py
```

Build the eval-style subset:

```bash
python scripts/data/build_paper_eval_subset.py
```

Build and download the eval image manifest:

```bash
python scripts/data/build_eval300_image_manifest.py
python scripts/data/download_viscounth_images_eval300.py
```

Prepare AER input/evidence:

```bash
python scripts/experiments/run_aer_prepare_eval300.py
```

## Generated Files

Common local generated files include:

- `data/processed/viscounth/viscounth_en_merged.csv`
- `data/processed/viscounth/viscounth_en_eval300.csv`
- `data/processed/viscounth/viscounth_en_eval300_manifest.csv`
- `data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv`
- `outputs/aer/viscounth_en_eval300_prepared.csv`

These files are reproducible artifacts and should not be committed.
