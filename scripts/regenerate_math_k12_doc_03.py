#!/usr/bin/env python3
"""
Regeneration script to fix mathematics K-12 doc 03 with full question stems.
This script regenerates only the mathematics K-12 document 03.
"""

import sys
from pathlib import Path

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging
from src.data_processing.dataset_downloader import DatasetDownloader
from src.pdf_generation.pdf_compiler import PDFCompiler
from main import (
    DatasetPool,
    build_domain_mcq,
    build_domain_tf,
    build_domain_long,
    render_latex,
    total_marks,
    save_json,
    generate_structured_json,
    get_fixed_question_distribution,
    infer_mcq_source_from_combination,
)

def regenerate_math_k12_doc_03():
    """Regenerate mathematics K-12 document 03 with fixed MCQ question formatting."""
    config = get_config()
    logger = get_logger()
    
    # Setup logging
    logging_config = config.get_logging_config()
    setup_logging(logging_config)
    
    domain = "mathematics"
    level = "K-12"
    doc_id = "mathematics_k-12_doc_03"
    
    logger.info(f"Regenerating {doc_id} with full question stems...")
    
    # Get combination for mathematics K-12
    combination = config.get_hierarchical_combination(domain, level)
    if not combination:
        logger.error(f"No combination found for {domain} {level}")
        return False
    
    # Get subjects
    subjects = config.get_subjects_for_domain_level(domain, level)
    if not subjects:
        logger.error(f"No subjects found for {domain} {level}")
        return False
    
    # Create output directories
    base_output = Path("output") / domain / level.lower()
    level_latex_dir = base_output / "latex_documents"
    level_pdf_dir = base_output / "pdf_documents"
    metadata_dir = Path("data/metadata_hierarchical")
    gold_dir = Path("data/gold_labels_hierarchical")
    json_output_dir = base_output / "JSON_output"
    
    level_latex_dir.mkdir(parents=True, exist_ok=True)
    level_pdf_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset pool
    pool = DatasetPool(seed=42, domain=domain, level=level, config=config)
    
    # Select combination (use first one for consistency)
    import random
    combo_rng = random.Random(42)
    selected_combination = combo_rng.choice(combination)
    mcq_source = infer_mcq_source_from_combination(selected_combination)
    has_native_long = any(qtype.endswith("_long") for qtype in selected_combination)
    
    # Get fixed question distribution
    question_distribution = get_fixed_question_distribution()
    
    # Build questions
    sections = {}
    
    # Build MCQ
    if question_distribution.get("mcq", 0) > 0:
        count = question_distribution["mcq"]
        sections["mcq"] = build_domain_mcq(pool, count, doc_id, domain)
    
    # Build True/False
    if question_distribution.get("tf", 0) > 0:
        count = question_distribution["tf"]
        sections["tf"] = build_domain_tf(pool, count, doc_id, domain)
    
    # Build Long-form
    if question_distribution.get("long", 0) > 0:
        count = question_distribution["long"]
        sections["long"] = build_domain_long(
            pool,
            count,
            doc_id,
            domain,
            mcq_source=mcq_source,
            use_native=has_native_long,
            allow_synthetic=True,
        )
    
    # Generate document
    marks = total_marks(sections)
    title = f"{domain.replace('_', ' ').title()} - {level} Level Assessment 3"
    latex_content = render_latex(title, marks, sections)
    latex_file = level_latex_dir / f"{doc_id}.tex"
    latex_file.write_text(latex_content, encoding="utf-8")
    
    # Compile PDF
    compiler = PDFCompiler(config)
    pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=level_pdf_dir)
    
    # Generate gold labels
    gold = {
        "document_id": doc_id,
        "domain": domain,
        "academic_level": level,
        "total_marks": marks,
        "subjects": subjects,
        "combination_used": selected_combination,
        "answers": [
            {
                "question_id": q.qid,
                "type": q.qtype,
                "correct_answer": q.correct_answer,
                "marks": q.marks,
                "explanation": q.explanation,
                "source_dataset": q.source_dataset,
                "source_id": q.source_id,
            }
            for questions in sections.values()
            for q in questions
        ],
    }
    save_json(gold, gold_dir / f"{doc_id}_gold.json")
    
    # Generate metadata
    metadata = {
        "document_id": doc_id,
        "domain": domain,
        "academic_level": level,
        "title": title,
        "total_marks": marks,
        "subjects": subjects,
        "combination_used": selected_combination,
        "latex_file": str(latex_file),
        "pdf_file": str(pdf_file),
        "generated_at": "2023-01-01",
        "version": "1.0"
    }
    save_json(metadata, metadata_dir / f"{doc_id}_metadata.json")
    
    # Generate structured JSON output
    structured_json = generate_structured_json(
        doc_id=doc_id,
        title=title,
        domain=domain,
        level=level,
        subjects=subjects,
        combination_used=selected_combination,
        sections=sections,
        gold_data=gold,
        metadata=metadata,
        latex_file=latex_file,
        pdf_file=pdf_file,
        compiler=compiler
    )
    save_json(structured_json, json_output_dir / f"{doc_id}.json")
    
    logger.info(f"Successfully regenerated {doc_id}")
    logger.info(f"PDF: {pdf_file}")
    logger.info(f"LaTeX: {latex_file}")
    
    return True

if __name__ == "__main__":
    success = regenerate_math_k12_doc_03()
    sys.exit(0 if success else 1)

