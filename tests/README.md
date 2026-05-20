# Tests

The tests in this directory are CPU-only and do not require VISCOUNTH files, downloaded images, model weights, or GPU access.

Run:

```bash
pytest
```

Current coverage focuses on deterministic AER support logic:

- prediction normalization
- template and keyword routing
- TF-IDF sentence retrieval on toy text
- template-aware answer extraction on toy evidence
