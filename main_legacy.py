#!/usr/bin/env python3
"""
Main execution script for IntegrityShield Data Preprocessing to PDF Generation System.

This script orchestrates the complete pipeline:
1. Download datasets from specified sources
2. Generate questions and documents
3. Create LaTeX documents
4. Compile to PDF
5. Apply IntegrityShield perturbations
6. Generate gold labels

Usage:
    python main.py [--config CONFIG_FILE] [--batch-size BATCH_SIZE] [--skip-download]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging, get_logger, log_success, log_error
from src.utils.config import get_config, ConfigManager
from src.data_processing.dataset_downloader import DatasetDownloader
from src.data_processing.question_generator import QuestionGenerator, QuestionType
from src.pdf_generation.latex_templates import LaTeXTemplates
from src.pdf_generation.pdf_compiler import PDFCompiler
from src.pdf_generation.integrity_shield import IntegrityShield


class PDFGenerationPipeline:
    """Main pipeline for PDF generation from datasets."""
    
    def __init__(self, config_path: str = "config.yaml", force_download: bool = False):
        """Initialize the pipeline."""
        self.config = get_config(config_path)
        self.logger = get_logger()
        
        # Initialize components
        self.dataset_downloader = DatasetDownloader(self.config, force_download=force_download)
        self.force_download = force_download
        self.question_generator = QuestionGenerator(self.config)
        self.latex_templates = LaTeXTemplates(self.config)
        self.pdf_compiler = PDFCompiler(self.config)
        self.integrity_shield = IntegrityShield(self.config)
        
        # Get output directories
        self.output_dirs = self.config.get_output_dirs()
        
        self.logger.info("PDF Generation Pipeline initialized")
    
    def run_dataset_download(self) -> Dict[str, Any]:
        """Load all configured datasets."""
        self.logger.info("Starting dataset loading phase")
        
        try:
            results = self.dataset_downloader.download_all_datasets()
            
            # Log results
            successful_downloads = sum(1 for r in results.values() if "error" not in r)
            total_datasets = len(results)
            
            log_success(f"Dataset loading completed", f"{successful_downloads}/{total_datasets} datasets loaded successfully")
            
            return results
            
        except Exception as e:
            log_error(e, "Dataset loading phase")
            raise
    
    def run_question_generation(self, batch_size: int = 5) -> List[Dict]:
        """Generate questions and documents."""
        self.logger.info(f"Starting question generation phase (batch size: {batch_size})")
        
        try:
            # Generate initial batch
            documents = self.question_generator.generate_initial_batch()
            
            log_success(f"Question generation completed", f"Generated {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            log_error(e, "Question generation phase")
            raise
    
    def run_latex_generation(self, documents: List[Dict]) -> List[Dict]:
        """Generate LaTeX documents and gold labels."""
        self.logger.info("Starting LaTeX generation phase")
        
        try:
            latex_output_dir = Path(self.output_dirs['latex_dir'])
            gold_output_dir = Path(self.output_dirs['gold_labels_dir'])
            
            # Process all documents
            results = self.latex_templates.process_multiple_documents(
                documents, latex_output_dir, gold_output_dir
            )
            
            log_success(f"LaTeX generation completed", f"Generated {len(results)} LaTeX documents and gold labels")
            
            return results
            
        except Exception as e:
            log_error(e, "LaTeX generation phase")
            raise
    
    def run_pdf_compilation(self, latex_files: List[Path]) -> List[Path]:
        """Compile LaTeX documents to PDF."""
        self.logger.info("Starting PDF compilation phase")
        
        try:
            pdf_files = self.pdf_compiler.compile_multiple_latex_files(latex_files)
            
            # Validate PDFs
            validation_results = self.pdf_compiler.batch_validate_pdfs(pdf_files)
            valid_pdfs = sum(validation_results.values())
            
            log_success(f"PDF compilation completed", f"Compiled {valid_pdfs}/{len(pdf_files)} valid PDFs")
            
            return pdf_files
            
        except Exception as e:
            log_error(e, "PDF compilation phase")
            raise
    
    def run_pdf_validation(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """Validate generated PDFs."""
        
        try:
            validation_results = self.pdf_compiler.batch_validate_pdfs(pdf_files)
            valid_count = sum(validation_results.values())
            total_count = len(pdf_files)
            
            log_success(f"PDF validation completed", f"{valid_count}/{total_count} PDFs are valid")
            
            return {
                "validation_results": validation_results,
                "valid_count": valid_count,
                "total_count": total_count,
                "valid_percentage": (valid_count / total_count) * 100 if total_count > 0 else 0
            }
            
        except Exception as e:
            log_error(e, "PDF validation phase")
            raise

    def run_integrity_shield(self, pdf_files: List[Path], documents: List[Any]) -> Dict[str, Any]:
        """Apply IntegrityShield perturbations to generated PDFs."""
        if not getattr(self.integrity_shield, 'enabled', False):
            self.logger.info("IntegrityShield disabled; skipping perturbation phase")
            return {
                "status": "disabled",
                "processed_documents_count": 0,
                "results": []
            }
        
        try:
            results = self.integrity_shield.process_documents(pdf_files, documents)
            log_success("IntegrityShield perturbations", f"Processed {len(results)} documents")
            return {
                "status": "completed",
                "processed_documents_count": len(results),
                "results": results
            }
        except Exception as e:
            log_error(e, "IntegrityShield phase")
            raise


    def run_complete_pipeline(self, skip_download: bool = False, batch_size: int = 5) -> Dict[str, Any]:
        """Run the complete PDF generation pipeline."""
        self.logger.info("Starting complete PDF generation pipeline")
        start_time = time.time()
        
        pipeline_results = {
            "start_time": start_time,
            "phases": {},
            "status": "running"
        }
        
        try:
            # Phase 1: Dataset Loading
            if not skip_download:
                self.logger.info("=== Phase 1: Dataset Loading ===")
                dataset_results = self.run_dataset_download()
                pipeline_results["phases"]["dataset_loading"] = {
                    "status": "completed",
                    "results": dataset_results
                }
            else:
                self.logger.info("Skipping dataset loading phase")
                pipeline_results["phases"]["dataset_loading"] = {
                    "status": "skipped",
                    "results": {}
                }
            
            # Phase 2: Question Generation
            self.logger.info("=== Phase 2: Question Generation ===")
            documents = self.run_question_generation(batch_size)
            pipeline_results["phases"]["question_generation"] = {
                "status": "completed",
                "documents_count": len(documents),
                "documents": documents
            }
            
            # Phase 3: LaTeX Generation
            self.logger.info("=== Phase 3: LaTeX Generation ===")
            latex_results = self.run_latex_generation(documents)
            pipeline_results["phases"]["latex_generation"] = {
                "status": "completed",
                "latex_files_count": len(latex_results),
                "latex_files": [str(r["latex_file"]) for r in latex_results]
            }
            
            # Phase 4: PDF Compilation
            self.logger.info("=== Phase 4: PDF Compilation ===")
            latex_files = [Path(r["latex_file"]) for r in latex_results]
            pdf_files = self.run_pdf_compilation(latex_files)
            pipeline_results["phases"]["pdf_compilation"] = {
                "status": "completed",
                "pdf_files_count": len(pdf_files),
                "pdf_files": [str(f) for f in pdf_files]
            }
            
            # Phase 5: PDF Validation
            self.logger.info("=== Phase 5: PDF Validation ===")
            validation_results = self.run_pdf_validation(pdf_files)
            pipeline_results["phases"]["pdf_validation"] = {
                "status": "completed",
                "validation_results": validation_results
            }
            
            # Phase 6: IntegrityShield Perturbations
            self.logger.info("=== Phase 6: IntegrityShield Perturbations ===")
            integrity_results = self.run_integrity_shield(pdf_files, documents)
            pipeline_results["phases"]["integrity_shield"] = {
                "status": integrity_results.get("status", "disabled"),
                "processed_documents_count": integrity_results.get("processed_documents_count", 0),
                "results": integrity_results.get("results", [])
            }

            # Pipeline completion
            end_time = time.time()
            duration = end_time - start_time
            
            pipeline_results.update({
                "end_time": end_time,
                "duration_seconds": duration,
                "status": "completed"
            })
            
            log_success("Complete PDF generation pipeline execution", f"Completed in {duration:.2f} seconds")
            
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            pipeline_results.update({
                "end_time": end_time,
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e)
            })
            
            log_error(e, "Complete PDF generation pipeline execution")
            
            # Save failed pipeline results
            self._save_pipeline_results(pipeline_results)
            
            raise
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        results_file = Path(self.output_dirs['logs_dir']) / "pipeline_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline results saved to: {results_file}")
    
    def generate_summary_report(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate a summary report of the pipeline execution."""
        report = []
        report.append("=" * 60)
        report.append("INTEGRITYSHIELD PIPELINE EXECUTION SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        status = pipeline_results.get("status", "unknown")
        duration = pipeline_results.get("duration_seconds", 0)
        report.append(f"Status: {status.upper()}")
        report.append(f"Duration: {duration:.2f} seconds")
        report.append("")
        
        # Phase results
        phases = pipeline_results.get("phases", {})
        for phase_name, phase_data in phases.items():
            report.append(f"Phase: {phase_name.replace('_', ' ').title()}")
            report.append(f"  Status: {phase_data.get('status', 'unknown')}")
            
            if phase_name == "dataset_loading":
                results = phase_data.get("results", {})
                successful = sum(1 for r in results.values() if "error" not in r)
                total = len(results)
                report.append(f"  Datasets: {successful}/{total} downloaded successfully")
            
            elif phase_name == "question_generation":
                count = phase_data.get("documents_count", 0)
                report.append(f"  Documents: {count} generated")
            
            elif phase_name == "latex_generation":
                count = phase_data.get("latex_files_count", 0)
                report.append(f"  LaTeX files: {count} generated")
            
            elif phase_name == "pdf_compilation":
                count = phase_data.get("pdf_files_count", 0)
                report.append(f"  PDF files: {count} compiled")
            
            elif phase_name == "integrity_shield":
                count = phase_data.get("processed_documents_count", 0)
                report.append(f"  Processed documents: {count}")
            
            report.append("")
        
        # Output directories
        report.append("Output Directories:")
        for dir_name, dir_path in self.output_dirs.items():
            report.append(f"  {dir_name}: {dir_path}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="IntegrityShield Data Preprocessing to PDF Generation Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of documents to generate in initial batch")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download phase")
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download of datasets even if cached")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PDFGenerationPipeline(args.config, force_download=args.refresh_data)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            skip_download=args.skip_download,
            batch_size=args.batch_size
        )
        
        # Generate and print summary report
        summary = pipeline.generate_summary_report(results)
        print(summary)
        
        # Save summary to file
        summary_file = Path(pipeline.output_dirs['logs_dir']) / "pipeline_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"\nSummary report saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
