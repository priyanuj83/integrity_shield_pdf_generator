"""
PDF compiler for IntegrityShield system.
Compiles LaTeX documents to PDF and handles PDF processing.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

from ..utils.logger import get_logger, log_success, log_error
from ..utils.config import get_config


class PDFCompiler:
    """Compiles LaTeX documents to PDF and manages PDF processing."""
    
    def __init__(self, config_manager=None):
        """Initialize PDF compiler."""
        self.config = config_manager or get_config()
        self.logger = get_logger()
        self.latex_output_dir = Path(self.config.get_output_dirs()['latex_dir'])
        self.pdf_output_dir = Path(self.config.get_output_dirs()['pdf_dir'])
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for LaTeX installation
        self._check_latex_installation()
    
    def _check_latex_installation(self):
        """Check if LaTeX is installed and available."""
        try:
            result = subprocess.run(['pdflatex', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info("LaTeX installation found")
            else:
                raise RuntimeError("LaTeX not properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError("LaTeX (pdflatex) not found. Please install LaTeX distribution.")
    
    def compile_latex_to_pdf(self, latex_file: Path, output_dir: Path = None) -> Path:
        """Compile LaTeX file to PDF."""
        if output_dir is None:
            output_dir = self.pdf_output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Compiling LaTeX to PDF: {latex_file}")
        
        try:
            # Create temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy LaTeX file to temp directory
                temp_latex = temp_path / latex_file.name
                shutil.copy2(latex_file, temp_latex)
                
                # Copy any required assets (like images)
                if latex_file.parent.exists():
                    for asset in latex_file.parent.glob("*"):
                        if asset.is_file() and asset.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf']:
                            shutil.copy2(asset, temp_path)
                
                # Compile LaTeX
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', str(temp_latex)],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    self.logger.error(f"LaTeX compilation failed: {result.stderr}")
                    raise RuntimeError(f"LaTeX compilation failed: {result.stderr}")
                
                # Move PDF to output directory
                pdf_file = output_dir / f"{latex_file.stem}.pdf"
                temp_pdf = temp_path / f"{temp_latex.stem}.pdf"
                
                if temp_pdf.exists():
                    shutil.move(str(temp_pdf), str(pdf_file))
                    log_success(f"PDF compilation: {latex_file.name}", f"Generated {pdf_file}")
                    return pdf_file
                else:
                    raise RuntimeError("PDF file not generated")
                    
        except subprocess.TimeoutExpired:
            raise RuntimeError("LaTeX compilation timed out")
        except Exception as e:
            log_error(e, f"Compiling LaTeX {latex_file}")
            raise
    
    def compile_multiple_latex_files(self, latex_files: List[Path], output_dir: Path = None) -> List[Path]:
        """Compile multiple LaTeX files to PDF."""
        self.logger.info(f"Compiling {len(latex_files)} LaTeX files to PDF")
        
        pdf_files = []
        for latex_file in latex_files:
            try:
                pdf_file = self.compile_latex_to_pdf(latex_file, output_dir)
                pdf_files.append(pdf_file)
            except Exception as e:
                self.logger.error(f"Failed to compile {latex_file}: {e}")
        
        log_success(f"Batch PDF compilation", f"Compiled {len(pdf_files)}/{len(latex_files)} files")
        return pdf_files
    
    def compile_document_directory(self, latex_dir: Path, pdf_dir: Path = None) -> List[Path]:
        """Compile all LaTeX files in a directory."""
        if pdf_dir is None:
            pdf_dir = self.pdf_output_dir
        
        latex_files = list(latex_dir.glob("*.tex"))
        self.logger.info(f"Found {len(latex_files)} LaTeX files in {latex_dir}")
        
        return self.compile_multiple_latex_files(latex_files, pdf_dir)
    
    def get_pdf_info(self, pdf_file: Path) -> Dict[str, Any]:
        """Get information about a PDF file."""
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
        file_size = pdf_file.stat().st_size
        
        # Try to get page count using pdfinfo if available
        page_count = None
        try:
            result = subprocess.run(['pdfinfo', str(pdf_file)], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Pages:'):
                        page_count = int(line.split(':')[1].strip())
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pdfinfo not available, use default
            page_count = 1
        
        return {
            "file_path": str(pdf_file),
            "file_size": file_size,
            "page_count": page_count or 1,
            "exists": True
        }
    
    def validate_pdf(self, pdf_file: Path) -> bool:
        """Validate that a PDF file is properly formatted."""
        try:
            # Try to get basic info
            info = self.get_pdf_info(pdf_file)
            
            # Check file size (should be > 0)
            if info['file_size'] <= 0:
                return False
            
            # Check if file can be opened (basic validation)
            with open(pdf_file, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"PDF validation failed for {pdf_file}: {e}")
            return False
    
    def batch_validate_pdfs(self, pdf_files: List[Path]) -> Dict[str, bool]:
        """Validate multiple PDF files."""
        results = {}
        
        for pdf_file in pdf_files:
            results[str(pdf_file)] = self.validate_pdf(pdf_file)
        
        valid_count = sum(results.values())
        self.logger.info(f"PDF validation: {valid_count}/{len(pdf_files)} files valid")
        
        return results
    
    def cleanup_auxiliary_files(self, latex_dir: Path):
        """Clean up LaTeX auxiliary files."""
        auxiliary_extensions = ['.aux', '.log', '.out', '.toc', '.fdb_latexmk', '.fls', '.synctex.gz']
        
        for ext in auxiliary_extensions:
            for file in latex_dir.glob(f"*{ext}"):
                try:
                    file.unlink()
                    self.logger.debug(f"Removed auxiliary file: {file}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {file}: {e}")


def main():
    """Main function for testing PDF compiler."""
    from ..utils.logger import setup_logging
    
    # Setup logging
    config = get_config()
    setup_logging(config.get_logging_config())
    
    # Initialize compiler
    compiler = PDFCompiler()
    
    # Test with sample LaTeX files
    latex_dir = Path("output/latex_documents")
    if latex_dir.exists():
        pdf_files = compiler.compile_document_directory(latex_dir)
        
        print(f"\nCompiled {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            info = compiler.get_pdf_info(pdf_file)
            print(f"- {pdf_file.name}: {info['file_size']} bytes, {info['page_count']} pages")
    else:
        print("No LaTeX files found to compile")


if __name__ == "__main__":
    main()
