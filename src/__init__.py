"""
IntegrityShield: Data Preprocessing to PDF Generation System

A comprehensive system for generating academic assessment documents
with IntegrityShield document-layer perturbations to prevent LLM-assisted cheating.
"""

__version__ = "1.0.0"
__author__ = "IntegrityShield Team"
__email__ = "integrity-shield@domain.com"

# Import main components for easy access
from .data_processing.dataset_downloader import DatasetDownloader
from .data_processing.question_generator import QuestionGenerator, QuestionType
from .pdf_generation.latex_templates import LaTeXTemplates
from .pdf_generation.pdf_compiler import PDFCompiler
from .pdf_generation.integrity_shield import IntegrityShield
from .utils.logger import get_logger, setup_logging
from .utils.config import get_config

__all__ = [
    "DatasetDownloader",
    "QuestionGenerator", 
    "QuestionType",
    "LaTeXTemplates",
    "PDFCompiler",
    "IntegrityShield",
    "get_logger",
    "setup_logging",
    "get_config"
]
