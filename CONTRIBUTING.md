# Contributing

Contributions are welcome once the project license and paper citation are finalized.

## Development Setup

```bash
conda env create -f environment.yml
conda activate heritage_workshop
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Expectations

- Do not commit raw datasets, processed data, predictions, logs, generated figures, or model weights.
- Keep scripts runnable from the repository root.
- Keep changes to model behavior explicit and documented.
- Add CPU-only tests for changes to routing, retrieval, extraction, or normalization logic.
- Run `python -m compileall src scripts` and `pytest` before opening a pull request.

## Reporting Issues

When reporting a reproducibility issue, include the command, model ID, dependency versions, GPU details if relevant, and whether the data files were regenerated locally.
