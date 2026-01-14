# Domain-Level Paper Generation Summary

## Overview

The IntegrityShield pipeline generates domain-aware assessment papers by sampling
from curated academic datasets:

- **MMLU / MMLU-Pro** – concept questions spanning K‑12 through graduate depth.
- **GSM8K** – structured long-form mathematics problems.
- **MBPP+** – programming challenges for CS-aligned domains.
- **AI2-ARC** – science K‑12 questions.

Domain-level configuration (see `config.yaml`) pairs each academic level with
dataset combinations. The generator then builds LaTeX, PDF, metadata, and gold
label artefacts that we can use as a research dataset.

## Counting Method

Counts were extracted from the repository structure
`output/<domain>/<level>/pdf_documents/`.  Temporary `test_run` artefacts were
excluded so only “final” dataset deliverables are represented.

### Snapshot Statistics

- **Domains covered:** 20
- **Total PDFs:** 135
- **Education level mix:**  
  - K‑12: 55 papers  
  - Undergraduate: 56 papers  
  - Graduate: 79 papers

### Papers per Domain and Level

| Domain | K‑12 | Undergraduate | Graduate | Total |
| --- | ---: | ---: | ---: | ---: |
| astronomy | 0 | 0 | 5 | 5 |
| biology | 2 | 2 | 5 | 9 |
| business | 1 | 2 | 5 | 8 |
| chemistry | 0 | 2 | 5 | 7 |
| computer_science | 6 | 5 | 5 | 16 |
| cybersecurity | 0 | 2 | 1 | 3 |
| economics | 6 | 5 | 5 | 16 |
| engineering | 1 | 5 | 5 | 11 |
| geography | 5 | 2 | 2 | 9 |
| health | 5 | 2 | 5 | 12 |
| history | 5 | 5 | 5 | 15 |
| machine_learning | 0 | 2 | 2 | 4 |
| mathematics | 5 | 4 | 5 | 14 |
| philosophy | 0 | 5 | 5 | 10 |
| physics | 0 | 2 | 5 | 7 |
| political_science | 0 | 2 | 1 | 3 |
| psychology | 0 | 5 | 5 | 10 |
| religious_studies | 5 | 0 | 0 | 5 |
| science | 5 | 0 | 0 | 5 |
| sociology | 0 | 2 | 2 | 4 |

### Papers by Education Level (all domains)

| Level | Count | Notable domains |
| --- | ---: | --- |
| K‑12 | 55 | heavy coverage in `science`, `religious_studies`, `computer_science`, `economics`, `history` |
| Undergraduate | 56 | balanced distribution; `mathematics` currently has 4 (one short of the target 5) |
| Graduate | 79 | broad coverage, with partial domains (`cybersecurity`, `political_science`) flagging follow-up work |

## Research Dataset Notes

- Each PDF is accompanied by structured JSON (metadata + gold labels) capturing
  question text, options, mark allocations, and dataset provenance.
- Counts reveal partial coverage in select graduate/undergraduate bands
  (e.g. cybersecurity graduate has 1 paper), highlighting potential targets for
  future generation runs.
- The artefacts can be reused for empirical studies (e.g. automated grading,
  domain adaptation, curriculum alignment) by referencing the domain/level
  folders documented above.



