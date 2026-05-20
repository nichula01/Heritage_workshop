# Source Package

The reusable code lives under `src/`, with AER-specific modules in `src/aer/`.

## Top-Level Modules

- `dataset_registry.py`: lightweight registry for dataset notes and root hints.
- `utils.py`: small shared utilities for seeds, directories, and JSON output.

## AER Modules

- `aer/router.py`: template-map and keyword fallback routing policy.
- `aer/retriever.py`: TF-IDF sentence splitting and retrieval.
- `aer/answer_extractor.py`: template-aware answer extraction rules.
- `aer/prediction_normalizer.py`: short-answer normalization utilities.
- `aer/pipeline.py`: evidence preparation wrapper combining router and retriever.
- `aer/internvl_vlm.py`: InternVL inference wrapper.
- `aer/minicpm_vlm.py`: MiniCPM inference wrapper.
- `aer/qwen_vlm.py`: Qwen inference wrapper.

CPU-only logic is covered by tests in `tests/`. VLM wrappers are optional and require model dependencies, checkpoint access, and suitable hardware.
