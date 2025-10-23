# IntegrityShield Data Preprocessing to PDF Generation System

## Project Structure

```
integrity_shield/
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── dataset_downloader.py
│   │   ├── question_generator.py
│   │   └── data_validator.py
│   ├── pdf_generation/
│   │   ├── __init__.py
│   │   ├── latex_templates.py
│   │   ├── pdf_compiler.py
│   │   └── integrity_shield.py
│   ├── testing/
│   │   ├── __init__.py
│   │   ├── test_data_processing.py
│   │   ├── test_pdf_generation.py
│   │   └── test_integrity_shield.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── gold_labels/
├── output/
│   ├── latex_documents/
│   ├── pdf_documents/
│   └── perturbed_documents/
├── templates/
│   ├── base_template.tex
│   ├── mcq_template.tex
│   ├── tf_template.tex
│   └── long_form_template.tex
├── tests/
├── logs/
├── requirements.txt
├── config.yaml
└── README.md
```

## Task Breakdown

### Stage 1: Data Downloading and Processing
- Download datasets from specified sources
- Process and validate data quality
- Create structured data format

### Stage 2: LaTeX Document Generation
- Generate 40 documents with different question type combinations
- Create 5 documents initially as requested
- Implement IntegrityShield perturbations

### Stage 3: Gold Label Generation
- Create answer keys for all documents
- Store in same template format as LaTeX documents

## Document Types to Generate (40 total)

1. MCQ + T/F + Long — 5 documents
2. MCQ + T/F — 5 documents  
3. MCQ + Long — 5 documents
4. MCQ — 5 documents
5. T/F + Long — 5 documents
6. T/F — 5 documents
7. Long — 5 documents
8. Mixed combinations — 5 documents

## Question Distribution per Document
- 5 × 2 = 10 marks
- 10 × 2 = 20 marks  
- 15 × 2 = 30 marks
- Total: 40 marks per document
