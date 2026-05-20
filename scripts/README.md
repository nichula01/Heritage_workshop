# Scripts

Run scripts from the repository root unless a script says otherwise.

## `scripts/data/`

- `check_viscounth_layout.sh`: print the expected VISCOUNTH raw folder layout.
- `inspect_viscounth_archives.py`: inspect VISCOUNTH zip archive members.
- `extract_viscounth_english.py`: extract English training and description archives.
- `inspect_viscounth_csvs.py`: preview extracted CSV schemas.
- `build_viscounth_subset.py`: merge English questions with descriptions and route labels.
- `build_paper_eval_subset.py`: build the balanced eval-style subset.
- `build_image_manifest.py`: build a small debug image manifest.
- `build_eval300_image_manifest.py`: build the eval-subset image manifest.
- `download_viscounth_images_small.py`: download small debug images.
- `download_viscounth_images_eval300.py`: download eval-subset images.
- `prepare_eval300_aer.py`: convert eval rows to JSONL for AER-style model input.
- `list_datasets.py`: print the lightweight dataset registry.

## `scripts/baselines/`

- `run_baseline.py`: placeholder baseline scaffold.
- `run_internvl_image_only.py`: InternVL image-only debug run.
- `run_internvl_image_only_eval300.py`: InternVL image-only eval run.
- `run_internvl_text_only.py`: InternVL text-only debug run.
- `run_internvl_text_only_eval300.py`: InternVL text-only eval run.
- `run_internvl_naive_multimodal.py`: InternVL naive image-plus-text debug run.
- `run_internvl_naive_multimodal_eval300.py`: InternVL naive image-plus-text eval run.
- `run_minicpm_image_only.py`: MiniCPM image-only debug run.

## `scripts/experiments/`

- `run_aer_prepare_subset.py`: prepare routed evidence for the debug subset.
- `run_aer_prepare_eval300.py`: prepare routed evidence for the eval subset.
- `run_internvl_routed_aer.py`: early routed AER debug run.
- `run_internvl_routed_aer_v2.py`: routed AER debug run with extraction.
- `run_internvl_routed_aer_v3.py`: routed AER debug run with updated policy behavior.
- `run_internvl_routed_aer_v3_eval300.py`: routed AER eval run.
- `run_internvl_aer_v4_eval300.py`: AER v4 eval run.
- `run_internvl_aer_v4_1_eval300.py`: AER v4.1 / TextFirst-style eval run.
- `run_internvl_aer_ablation_eval300.py`: configurable ablation runner.
- `run_minicpm_eval300.py`: MiniCPM eval JSONL runner.

## `scripts/evaluation/`

- `evaluate_predictions_basic.py`: compute exact-match and contains-match metrics.
- `analyze_baseline_winners_eval300.py`: derive a template policy from baseline winners.
- `eval_sample_size_robustness.py`: estimate robustness across sample sizes.

## `scripts/figures/`

- `plot_sample_size_robustness.py`: plot sample-size robustness curves from local results.

## `scripts/setup/`

- `setup_env.sh`: create or update the base conda environment.
- `setup_minicpm_env.sh`: create or update the MiniCPM environment.
- `setup_qwen_vlm.sh`: install Qwen VLM dependencies.
- `check_gpu_qwen.py`: print PyTorch/CUDA availability for Qwen runs.
- `run_internvl_smoke.py`: quick InternVL smoke test.
- `run_minicpm_smoke.py`: quick MiniCPM smoke test.
- `run_qwen_smoke.py`: quick Qwen smoke test.
