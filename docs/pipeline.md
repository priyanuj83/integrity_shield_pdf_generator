# Pipeline Walkthrough

This document expands on the six execution phases run by `PDFGenerationPipeline`.

## 1. Dataset Loading
- Uses `ConfigManager.get_dataset_config()` to resolve URLs or local paths.
- Hugging Face datasets are pulled via `datasets.load_dataset`; local GSM8K archives are parsed from JSONL files.
- Normalised JSON snapshots are stored under `data/raw/`.
- Results are cached so `--skip-download` can reuse existing payloads.

## 2. Question Generation
- `QuestionGenerator.generate_initial_batch()` reads cached datasets and produces a list of `Document` dataclasses.
- Question mixes are derived from `document_generation.combinations` in `config.yaml`.
- Generated documents are persisted as JSON under `data/processed/` for traceability.

## 3. LaTeX Rendering
- `LaTeXTemplates.process_multiple_documents()` converts each `Document` into LaTeX using the base template in `templates/`.
- Section builders handle MCQ, True/False, and long-form questions independently.
- Answer keys (gold labels) accompany each document and are written to `data/gold_labels/`.

## 4. PDF Compilation
- `PDFCompiler.compile_multiple_latex_files()` shells out to `pdflatex` inside a temporary directory per file.
- Supporting assets (images etc.) co-located with the LaTeX file are copied automatically.
- Failing compilations raise exceptions and are reported in the pipeline summary.

## 5. PDF Validation
- Runs lightweight checks on file existence, size, and PDF headers (leveraging `pdfinfo` when available).
- Aggregated results are written to `logs/pipeline_results.json`.

## Monitoring & Reports
- Log output streams to stdout and `logs/integrity_shield.log` via Loguru.
- `_save_pipeline_results` stores a machine-readable JSON summary.
- `PDFGenerationPipeline.generate_summary_report()` creates a text report (also persisted at `logs/pipeline_summary.txt`).
