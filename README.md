# Adaptive Evidence Routing for Cultural Heritage Visual Question Answering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-models-yellow)
![Research Code](https://img.shields.io/badge/Research-Code-green)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

This repository contains research code for **Adaptive Evidence Routing (AER)** for cultural heritage visual question answering. The project studies how to answer VISCOUNTH-style questions using the evidence source each question actually needs: the image, contextual metadata/description text, or a combination of image and retrieved text. The code is organized for lightweight reproduction without committing datasets, model weights, generated outputs, or private paper metadata.

## Method Diagram

`docs/assets/aer_overview.png` is coming soon. The intended diagram will show the question, image, metadata/description, template-aware router, TF-IDF evidence retrieval, VLM fallback, and short-answer normalization stages.

## Key Idea

Cultural heritage VQA is heterogeneous. Some questions are visual, such as color, shape, or visible objects. Some are contextual, such as author, period, owner, or location. Others need both the image and text evidence, such as inscriptions or depicted elements. AER routes each question to the evidence source it needs instead of sending every sample through one fixed multimodal prompt.

## Features

- VISCOUNTH English subset preparation and eval-subset utilities
- TF-IDF sentence retrieval over cultural-property descriptions
- Template-aware answer extraction for structured heritage question types
- Image-only, text-only, and naive multimodal baselines
- AER and AER-TextFirst style routed variants
- Ablation scripts for extraction, normalization, routing, and policy choices
- Sample-size robustness evaluation and plotting utilities
- Optional VLM wrappers for InternVL, MiniCPM, and Qwen backbones

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── CONTRIBUTING.md
├── configs/
│   └── base_config.yaml
├── data/
│   └── README.md
├── docs/
│   ├── LICENSE_NOTE.md
│   ├── dataset_setup.md
│   ├── experiment_guide.md
│   ├── literature_tracker.md
│   ├── project_overview.md
│   └── roadmap.md
├── metadata/
│   ├── dataset_notes.md
│   └── viscounth_route_map.json
├── notebooks/
│   └── README.md
├── paper/
│   ├── README.md
│   ├── abstract.md
│   └── outline.md
├── results/
│   └── README.md
├── scripts/
│   ├── README.md
│   ├── baselines/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── figures/
│   └── setup/
├── src/
│   ├── README.md
│   ├── dataset_registry.py
│   ├── utils.py
│   └── aer/
└── tests/
    └── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nichula01/heritage-workshop.git
cd heritage-workshop
```

2. Create the base conda environment:

```bash
conda env create -f environment.yml
conda activate heritage_workshop
```

3. Install the lightweight Python requirements:

```bash
python -m pip install -r requirements.txt
```

4. Install optional model-specific dependencies only when running VLM inference:

```bash
python -m pip install -r requirements-models.txt
```

For MiniCPM-specific experiments, see `environment_minicpm.yml` and `scripts/setup/setup_minicpm_env.sh`.

## Dataset Setup

VISCOUNTH data must be downloaded separately according to its source terms and license. Dataset files are intentionally ignored by Git.

Expected local layout:

```text
data/
├── raw/
│   ├── viscounth_repo/
│   ├── viscounth_extracted/
│   ├── viscounth_images_small/
│   └── viscounth_images_eval300/
└── processed/
    └── viscounth/
```

See [docs/dataset_setup.md](docs/dataset_setup.md) for the full setup notes.

## Quick Start

Inspect the expected VISCOUNTH layout:

```bash
bash scripts/data/check_viscounth_layout.sh
python scripts/data/inspect_viscounth_archives.py
```

Extract the English subset and build processed metadata:

```bash
python scripts/data/extract_viscounth_english.py
python scripts/data/build_viscounth_subset.py
python scripts/data/build_paper_eval_subset.py
```

Build image manifests and prepare AER inputs:

```bash
python scripts/data/build_eval300_image_manifest.py
python scripts/data/download_viscounth_images_eval300.py
python scripts/experiments/run_aer_prepare_eval300.py
```

Run baselines and routed AER variants:

```bash
python scripts/baselines/run_internvl_image_only_eval300.py
python scripts/baselines/run_internvl_text_only_eval300.py
python scripts/baselines/run_internvl_naive_multimodal_eval300.py
python scripts/experiments/run_internvl_aer_v4_1_eval300.py
```

Evaluate predictions:

```bash
python scripts/evaluation/evaluate_predictions_basic.py results/internvl25_2b_aer_v4_1_eval300/predictions.csv
```

All paths above are relative to the repository root. Generated outputs under `data/`, `outputs/`, and `results/` are local artifacts and are ignored by Git.

## Results

We report paper-level draft experiments on the English-only VISCOUNTH evaluation subset of 298 samples. The evaluation uses two short-answer metrics, both reported as percentages:

- **Exact Match (EM)**: prediction exactly matches the normalized gold answer.
- **Contains**: relaxed match where the gold answer is contained in the prediction or vice versa.

### Main comparison

| Method | EM ↑ | Contains ↑ |
|---|---:|---:|
| Image-only baseline | 8.72 | 10.07 |
| Text-only baseline | 37.25 | 45.97 |
| Naive multimodal baseline | 13.09 | 22.15 |
| Adaptive Evidence Routing | 38.26 | 51.34 |
| **AER-TextFirst** | **44.97** | **53.69** |

Brief interpretation:
AER-TextFirst gives the strongest overall performance. The text-only baseline is much stronger than the image-only baseline, showing that cultural heritage VQA often depends heavily on metadata and descriptions. However, naive image+full-description prompting performs poorly, showing that simply adding more context can distract the model. AER improves performance by routing each question to the evidence source it actually needs.

### Performance by question type

| Method | Visual EM ↑ | Contextual EM ↑ | Mixed EM ↑ |
|---|---:|---:|---:|
| Image-only baseline | 20.0 | 1.0 | 5.1 |
| Text-only baseline | 50.0 | 38.4 | 23.2 |
| Naive multimodal | 23.0 | 6.1 | 10.1 |
| AER | 50.0 | 34.3 | 30.3 |
| **AER-TextFirst** | **54.0** | **41.4** | **39.4** |

Brief interpretation:
The largest improvement appears on mixed questions, where AER-TextFirst reaches 39.4% EM compared with 10.1% for naive multimodal prompting. This supports the central hypothesis that cultural heritage questions require question-conditioned evidence selection rather than a fixed evidence strategy.

### Ablation study

| Variant | EM ↑ | Contains ↑ |
|---|---:|---:|
| **AER-TextFirst** | **44.97** | **53.69** |
| w/o extractor | 38.93 | 47.99 |
| w/o normalization | 42.47 | 52.65 |
| w/o template policy, global text-first | 44.63 | 53.69 |
| route-based fallback policy | 29.19 | 35.91 |
| retrieved-only extraction | 40.94 | 51.34 |
| full-description-only extraction | 44.30 | 53.36 |
| w/o SHAPE image-only override | 43.62 | 51.68 |

Brief interpretation:
The template-aware extractor is the most important component. Removing it drops EM from 44.97% to 38.93%. The text-first design is also important: a route-based fallback policy performs much worse, suggesting that answer extraction should be attempted before expensive or noisy VLM inference.

### Template-wise gains

| Template | Naive Multimodal EM ↑ | Routed + Retrieval EM ↑ |
|---|---:|---:|
| AFFIXEDAUTHOR | 25.0 | 100.0 |
| AUTHOR | 25.0 | 75.0 |
| DATING | 25.0 | 75.0 |
| AFFIXEDLANGUAGE | 37.5 | 75.0 |
| AFFIXEDTRANSCRIPT | 12.5 | 75.0 |
| AFFIXEDELEMENT | 0.0 | 62.5 |
| AFFIXEDTECHNIQUE | 0.0 | 62.5 |
| SUBJECT | 14.3 | 57.1 |

Brief interpretation:
AER is especially effective for structured metadata-heavy templates such as author, language, transcript, dating, and technique questions. These gains come mainly from retrieving focused textual evidence and applying template-aware answer extraction.

### Key takeaway

The results show that cultural heritage VQA is not only a multimodal reasoning problem, but also an evidence-selection problem. A fixed image-only, text-only, or naive multimodal strategy is suboptimal because different questions require different evidence sources. AER-TextFirst improves robustness by combining targeted text retrieval, template-aware extraction, policy-guided VLM fallback, and answer normalization.

> These numbers are from draft paper experiments and should be reproduced locally before being treated as final benchmark results.

## Method Overview

1. Retrieve top-k text evidence from the cultural-property description with TF-IDF sentence retrieval.
2. Apply template-aware extraction for question types that can be answered from structured text evidence.
3. Fall back to a policy-guided VLM mode: image-only, text-only, or image plus retrieved text.
4. Normalize the short answer for fairer exact-match and contains-match evaluation.

## Reproducibility Notes

- Default seed is `42` where scripts construct subsets or sample trials.
- The eval subset should be regenerated from local VISCOUNTH files and recorded with its stats JSON.
- VLM results depend on model checkpoint, quantization, prompt wrapper, hardware, and library versions.
- GPU-backed scripts are optional and should not be required for CPU-only tests.
- Keep generated predictions, logs, cached models, and downloaded data out of Git.

## Citation

Citation information is not finalized. See [CITATION.cff](CITATION.cff) for placeholders. If you use this code, please cite the associated paper once available.

## Acknowledgements

This project builds on the VISCOUNTH dataset and open VLM backbones available through the broader PyTorch and HuggingFace ecosystems, including InternVL, MiniCPM, and Qwen model families.

## Contact

- GitHub: [nichula01](https://github.com/nichula01)
- Email: e20425@eng.pdn.ac.lk
