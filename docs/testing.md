# Testing Strategy

## Automated Tests
- `pytest src/testing/test_data_processing.py`
  - Verifies dataset parsing, question generation, and helper utilities via mocks.
- `pytest src/testing/test_pdf_generation.py`
  - Covers LaTeX rendering, PDF compilation helpers, and IntegrityShield perturbation workflows.

> Install with `pip install pytest pytest-cov` if not already available.

## Manual Smoke Tests
- `python scripts/test_dataset_loading.py`
  - Exercises remote + local dataset loading; useful when updating `config.yaml` datasets.
- `python scripts/test_standalone.py`
  - Confirms required directories exist and `PDFGenerationPipeline` initialises cleanly.
- `python scripts/test_system.py`
  - Performs structural integrity checks and basic YAML validation.

## Troubleshooting
- **Missing PyMuPDF (`fitz`)**: Install `PyMuPDF` or allow tests to rely on mocks (default behaviour).
- **LaTeX failures**: Ensure `pdflatex` is in `PATH`. On macOS install MacTeX; on Ubuntu use `texlive-full`.
- **Dataset download errors**: Check network access or set `--skip-download` after the first successful run.

## Coverage Ideas
- Add regression tests for new perturbation types by extending `TestIntegrityShield`.
- Use `pytest --cov=src src/testing` to monitor coverage trends.
