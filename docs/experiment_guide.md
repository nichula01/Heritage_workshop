# Experiment Guide

Run commands from the repository root. Dataset files, predictions, and plots are local artifacts ignored by Git.

## Baselines

Image-only baseline:

```bash
python scripts/baselines/run_internvl_image_only_eval300.py
```

Text-only baseline:

```bash
python scripts/baselines/run_internvl_text_only_eval300.py
```

Naive multimodal baseline:

```bash
python scripts/baselines/run_internvl_naive_multimodal_eval300.py
```

The baseline scripts write predictions under `results/<run_name>/predictions.csv`.

## AER Preparation

Prepare routed evidence before running routed VLM experiments:

```bash
python scripts/experiments/run_aer_prepare_eval300.py \
  --input-csv data/processed/viscounth/viscounth_en_eval300_manifest_downloaded.csv \
  --output-csv outputs/aer/viscounth_en_eval300_prepared.csv \
  --top-k 2
```

## AER Variants

Routed AER variants:

```bash
python scripts/experiments/run_internvl_routed_aer_v3_eval300.py
python scripts/experiments/run_internvl_aer_v4_eval300.py
python scripts/experiments/run_internvl_aer_v4_1_eval300.py
```

Ablation entry point:

```bash
python scripts/experiments/run_internvl_aer_ablation_eval300.py --tag ablation_run
```

Use distinct `--tag` values so each run writes to a separate folder under `results/`.

## Evaluation

Basic exact-match and contains-match evaluation:

```bash
python scripts/evaluation/evaluate_predictions_basic.py results/internvl25_2b_aer_v4_1_eval300/predictions.csv
```

Build a template policy from baseline winners:

```bash
python scripts/evaluation/analyze_baseline_winners_eval300.py
```

Sample-size robustness:

```bash
python scripts/evaluation/eval_sample_size_robustness.py
python scripts/figures/plot_sample_size_robustness.py
```

## Model Notes

- InternVL, MiniCPM, and Qwen experiments require optional model dependencies and local GPU capacity.
- Model downloads should use local HuggingFace cache directories outside Git.
- Record model IDs, revisions, device type, and precision when reporting results.
