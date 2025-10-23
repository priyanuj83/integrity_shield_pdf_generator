# IntegrityShield Assessment Pipeline

IntegrityShield is a Python pipeline for generating assessment-ready PDFs from curated datasets. The system downloads or loads source data, synthesises multiple question types, renders LaTeX, compiles PDFs, and optionally applies document-layer perturbations to hinder automated answer scraping.

## Key Capabilities
- Dataset ingestion from local archives or Hugging Face repositories
- Question synthesis across MCQ, True/False, and long-form formats
- LaTeX templating with configurable exam metadata
- PDF compilation with auxiliary file cleanup
- Logging, summary reports, and gold-label generation for downstream evaluation

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: install PyMuPDF if you plan to run IntegrityShield perturbations
pip install PyMuPDF

# Generate assessment PDFs (downloads datasets on first run)
python main.py --skip-download --count 5

# Force a fresh dataset pull when needed
python main.py --refresh-data --count 5
```

### Core CLI Flags
`python main.py [--config CONFIG_YAML] [--count N] [--skip-download] [--refresh-data] [--verbose]`

- `--config`: alternate configuration file (defaults to `config.yaml`)
- `--count`: number of documents generated per run (default `5`); each document samples a combination from `document_generation.combinations`
- `--skip-download`: reuse existing datasets without refreshing
- `--refresh-data`: force a re-download of every dataset even if cached
- `--verbose`: stream additional log output to the console

## Repository Layout
```
integrity_shield/
├── config.yaml                  # Default configuration
├── main.py                      # Primary Gen2 generation CLI (this pipeline)
├── main_legacy.py               # Previous end-to-end pipeline (retained for reference)
├── requirements.txt             # Python dependencies
├── scripts/                     # Command-line helpers and smoke tests
├── src/                         # Application code
│   ├── data_processing/         # Dataset ingestion and question synthesis
│   ├── pdf_generation/          # LaTeX + PDF + IntegrityShield components
│   └── utils/                   # Configuration & logging helpers
├── templates/                   # LaTeX templates (tracked)
├── docs/                        # Documentation, references, archived notes
├── examples/                    # Sample artefacts for demonstration only
├── data/                        # Placeholder for datasets (ignored by Git)
├── output/                      # Generated LaTeX/PDF artefacts (ignored)
└── logs/                        # Runtime logs and summaries (ignored)
```

## Pipeline Overview
Each execution of `main.py` (or the helper scripts) follows the stages below:

1. **Dataset Loading** – pull configured datasets and normalise to JSON payloads.
2. **Question Generation** – sample MCQ / True-False / Long prompts to build document manifests.
3. **LaTeX Generation** – render printable assessments plus gold-label JSON answer keys.
4. **PDF Compilation** – compile LaTeX to PDF using `pdflatex`, reporting invalid builds.

Detailed explanations for each phase, configuration knobs, and extension points are available in `docs/` (see links below).

## Data & Output Hygiene
Generated datasets, build artefacts, and logs are ignored via `.gitignore`. The repository includes `.gitkeep` markers so `data/`, `output/`, and `logs/` remain in version control without shipping large artefacts. Drop new inputs into `data/` and inspect results in `output/` after running the pipeline.

## Useful Scripts
- `python scripts/generate_pdfs.py` – thin wrapper around `python main.py` for convenience.
- `python scripts/run_initial_batch.py` – wrapper for generating the first batch (default 5 documents).
- `python scripts/test_dataset_loading.py` – smoke-check dataset ingestion (requires network access for Hugging Face sources).
- `python scripts/test_standalone.py` / `python scripts/test_system.py` – quick structure checks for local development.

> ℹ️  Downstream perturbation or signature services are handled separately; this repository stops after PDF + gold-label generation.

## Testing
The project ships with `pytest` suites under `src/testing/` covering data processing and PDF generation components. Run them with:
```bash
pytest src/testing
```
If `pytest` is not installed globally, install it inside your virtual environment (`pip install pytest pytest-cov`). Some tests mock third-party dependencies (e.g., PyMuPDF) so they run even when optional packages are absent.

## Documentation
- `docs/README.md` – documentation index and navigation
- `docs/architecture.md` – component relationships and data flow diagrams in prose
- `docs/pipeline.md` – step-by-step walkthrough of the generation pipeline
- `docs/data_management.md` – guidance on datasets, caching, and storage hygiene
- `docs/testing.md` – testing strategy, fixtures, and troubleshooting tips
- `docs/archive/` – historical notes and previous implementation summaries

## Next Steps
- Configure additional datasets in `config.yaml` (`datasets:` section)
- Extend LaTeX templates in `templates/`
- Integrate pipeline runs with CI (see `scripts/test_system.py` for baseline checks)

---

> ℹ️  This repository currently has no explicit license. Add one before distributing or contributing externally.
