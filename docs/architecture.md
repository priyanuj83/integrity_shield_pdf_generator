# Architecture Overview

IntegrityShield is organised as a modular pipeline where each layer performs an isolated responsibility. The design keeps data ingestion, question synthesis, document rendering, and perturbations decoupled so each part can be extended independently.

## Components
- **`src/data_processing/`**
  - `dataset_downloader.py`: loads local JSONL archives or Hugging Face datasets, normalises them to JSON, and caches the results under `data/raw/`.
  - `question_generator.py`: builds `Document` objects composed of MCQ, T/F, and long-form questions derived from processed datasets.
- **`src/pdf_generation/`**
  - `latex_templates.py`: renders documents into LaTeX using configurable metadata and section builders.
  - `pdf_compiler.py`: compiles LaTeX to PDF, validates output, and cleans auxiliary build files.
  - `integrity_shield.py`: legacy perturbation utilities (kept for reference; not used by the Gen2 pipeline).
- **`src/utils/`**
  - `config.py`: YAML-backed configuration loader exposing structured dataclasses for each subsystem.
  - `logger.py`: centralised Loguru-based logging with console + rotating file outputs.
- **`main.py`**
  - `PDFGenerationPipeline`: orchestrates all phases, collects metrics, and persists summary reports.

## Data Flow
```
Datasets (local/hf) → DatasetDownloader → QuestionGenerator → LaTeXTemplates
      ↓                               ↑                ↓
  data/raw/*.json              data/processed/*.json   output/latex_documents
                                                        ↓
                                                   PDFCompiler
                                                        ↓
                                               output/pdf_documents
                                                        ↓
                                              IntegrityShield (optional)
                                                        ↓
                                            output/perturbed_documents + signatures
```

## Extensibility Hooks
- Add new dataset entries in `config.yaml` to extend question coverage.
- Provide alternative LaTeX templates under `templates/` and update `config.yaml:latex.base_template`.
- Integrate CI by invoking `main.main()` with custom configs or by leveraging scripts under `scripts/`.
