# Data Management Guide

## Directory Policy
- `data/` – only lightweight `.gitkeep` file is tracked; actual datasets are ignored to keep the repository lean.
  - `raw/`: normalised JSON dumps created by `DatasetDownloader`.
  - `processed/`: serialised `Document` objects used for LaTeX generation.
  - `gold_labels/`: answer keys paired with generated assessments.
- `output/` – stores generated LaTeX, compiled PDFs, and IntegrityShield artefacts. Everything is ignored by Git.
- `logs/` – Loguru log files and pipeline summaries (ignored).

## Cleaning Up
Use the helper below to reset working directories without touching tracked sources:
```bash
rm -rf data/raw/* data/processed/* data/gold_labels/*
rm -rf output/* logs/*
```

## Adding New Datasets
1. Place local archives (e.g., GSM8K JSONL) under `data/gsm_mcq/`.
2. Extend the `datasets:` section of `config.yaml` with the desired identifier and settings.
3. Run `python scripts/test_dataset_loading.py` to confirm the dataset loads successfully.

## Storage Tips
- Keep dataset snapshots small by limiting Hugging Face pulls to the splits you need.
- When sharing the project, avoid checking large artefacts into version control; rely on configuration to rebuild instead.
- Update `.gitignore` if you introduce new top-level caches.
