# Project Overview

Adaptive Evidence Routing (AER) is a lightweight routing approach for cultural heritage visual question answering. The method is designed for VISCOUNTH-style questions where the answer may come from visual evidence, contextual metadata, or both.

## Motivation

Cultural heritage collections contain rich metadata and images, but question types vary substantially. A single fixed prompt can waste context or use the wrong evidence source. AER makes the evidence choice explicit before answer generation.

## Pipeline

1. Read the question, template ID, image path, and textual description.
2. Predict an evidence route using the template map and keyword fallback rules.
3. Retrieve top-k description sentences with TF-IDF when text evidence is needed.
4. Try template-aware extraction for structured question types.
5. Use a VLM fallback when extraction is insufficient.
6. Normalize the predicted short answer for evaluation.

## Main Code Areas

- `src/aer/router.py`: template and keyword route policy.
- `src/aer/retriever.py`: TF-IDF sentence retrieval.
- `src/aer/answer_extractor.py`: template-aware extraction rules.
- `src/aer/pipeline.py`: evidence preparation wrapper.
- `src/aer/*_vlm.py`: optional VLM wrappers.
- `scripts/`: data preparation, baselines, experiments, evaluation, and figures.

## Scope

This repository is for reproducible research code. It intentionally excludes raw datasets, processed dataset files, predictions, generated result tables, logs, model weights, and non-public paper metadata.
