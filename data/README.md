# Data Directory

This directory is reserved for local dataset files. Do not commit raw data, processed CSV/JSONL files, downloaded images, archives, or cached artifacts.

Expected local folders:

- `raw/viscounth_repo/`: manually downloaded VISCOUNTH source files.
- `raw/viscounth_extracted/`: extracted English training and description files.
- `raw/viscounth_images_small/`: optional small image subset.
- `raw/viscounth_images_eval300/`: optional eval-subset images.
- `processed/viscounth/`: generated CSV/JSONL files and stats.

See `docs/dataset_setup.md` for the full workflow.
