"""
Tests for PDF generation components.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ..pdf_generation.latex_templates import LaTeXTemplates
from ..pdf_generation.pdf_compiler import PDFCompiler
from ..pdf_generation.integrity_shield import IntegrityShield
from ..utils.config import get_config


class TestLaTeXTemplates:
    """Test cases for LaTeX templates."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'latex_dir': self.temp_dir,
            'gold_labels_dir': self.temp_dir
        }
        self.config.get_latex_config.return_value = Mock(
            base_template="templates/base_template.tex",
            output_dir=self.temp_dir,
            pdf_output_dir=self.temp_dir,
            school_name="Test School",
            course_number="TEST-101",
            subject="Test Subject",
            term="2023"
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_mcq_section(self):
        """Test MCQ section generation."""
        templates = LaTeXTemplates(self.config)
        
        questions = [
            {
                "text": "What is 2+2?",
                "options": ["3", "4", "5", "6"],
                "marks": 2
            },
            {
                "text": "What is 3+3?",
                "options": ["5", "6", "7", "8"],
                "marks": 2
            }
        ]
        
        latex = templates.generate_mcq_section(questions)
        
        assert "\\section*{MCQ with Reason (4 marks)}" in latex
        assert "What is 2+2?" in latex
        assert "What is 3+3?" in latex
        assert "\\begin{enumerate}" in latex
        assert "\\end{enumerate}" in latex
    
    def test_generate_tf_section(self):
        """Test True/False section generation."""
        templates = LaTeXTemplates(self.config)
        
        questions = [
            {"text": "2+2 equals 4.", "marks": 2},
            {"text": "3+3 equals 7.", "marks": 2}
        ]
        
        latex = templates.generate_tf_section(questions)
        
        assert "\\section*{True/False with Reason (4 marks)}" in latex
        assert "2+2 equals 4." in latex
        assert "3+3 equals 7." in latex
        assert "Write True or False" in latex
    
    def test_generate_long_section(self):
        """Test long-form section generation."""
        templates = LaTeXTemplates(self.config)
        
        questions = [
            {"text": "Explain the concept of addition.", "marks": 5},
            {"text": "Describe the properties of numbers.", "marks": 5}
        ]
        
        latex = templates.generate_long_section(questions)
        
        assert "\\section*{Solve (10 marks)}" in latex
        assert "Explain the concept of addition." in latex
        assert "Describe the properties of numbers." in latex
        assert "Show all steps clearly" in latex
    
    def test_generate_document_latex(self):
        """Test complete document LaTeX generation."""
        templates = LaTeXTemplates(self.config)
        
        document_data = {
            "id": "test_doc",
            "total_marks": 20,
            "questions": [
                {
                    "type": "mcq",
                    "text": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "marks": 2
                },
                {
                    "type": "tf",
                    "text": "2+2 equals 4.",
                    "marks": 2
                }
            ]
        }
        
        latex = templates.generate_document_latex(document_data)
        
        assert "\\documentclass[12pt]{article}" in latex
        assert "Assessment - test_doc" in latex
        assert "Total Marks: 20" in latex
        assert "What is 2+2?" in latex
        assert "2+2 equals 4." in latex
    
    def test_generate_gold_labels(self):
        """Test gold labels generation."""
        templates = LaTeXTemplates(self.config)
        
        document_data = {
            "id": "test_doc",
            "total_marks": 20,
            "questions": [
                {
                    "id": "q1",
                    "type": "mcq",
                    "correct_answer": "4",
                    "marks": 2,
                    "explanation": "2+2=4"
                },
                {
                    "id": "q2",
                    "type": "tf",
                    "correct_answer": "True",
                    "marks": 2,
                    "explanation": "This is correct"
                }
            ]
        }
        
        gold_labels = templates.generate_gold_labels(document_data)
        
        assert gold_labels["document_id"] == "test_doc"
        assert gold_labels["total_marks"] == 20
        assert len(gold_labels["answers"]) == 2
        assert gold_labels["answers"][0]["correct_answer"] == "4"
        assert gold_labels["answers"][1]["correct_answer"] == "True"


class TestPDFCompiler:
    """Test cases for PDF compiler."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'latex_dir': self.temp_dir,
            'pdf_dir': self.temp_dir
        }
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_check_latex_installation_success(self, mock_run):
        """Test successful LaTeX installation check."""
        mock_run.return_value = Mock(returncode=0)
        
        compiler = PDFCompiler(self.config)
        
        # Should not raise exception
        assert compiler is not None
    
    @patch('subprocess.run')
    def test_check_latex_installation_failure(self, mock_run):
        """Test failed LaTeX installation check."""
        mock_run.side_effect = FileNotFoundError("LaTeX not found")
        
        with pytest.raises(RuntimeError, match="LaTeX.*not found"):
            PDFCompiler(self.config)
    
    def test_get_pdf_info(self):
        """Test PDF info extraction."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%dummy content")
        
        compiler = PDFCompiler(self.config)
        info = compiler.get_pdf_info(pdf_file)
        
        assert info["file_path"] == str(pdf_file)
        assert info["file_size"] > 0
        assert info["exists"] is True
    
    def test_validate_pdf_valid(self):
        """Test PDF validation for valid file."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%dummy content")
        
        compiler = PDFCompiler(self.config)
        is_valid = compiler.validate_pdf(pdf_file)
        
        assert is_valid is True
    
    def test_validate_pdf_invalid(self):
        """Test PDF validation for invalid file."""
        # Create an invalid file
        pdf_file = Path(self.temp_dir) / "test.txt"
        pdf_file.write_text("This is not a PDF")
        
        compiler = PDFCompiler(self.config)
        is_valid = compiler.validate_pdf(pdf_file)
        
        assert is_valid is False


class TestIntegrityShield:
    """Test cases for IntegrityShield."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'perturbed_dir': self.temp_dir
        }
        self.config.get_integrity_shield_config.return_value = Mock(
            enabled=True,
            perturbation_types=["hidden_text", "font_remapping", "visual_overlay"],
            hidden_text={"color": "white", "position_offset": [2, 3]},
            font_remapping={"unicode_mappings": {"CNN": "RNN"}},
            visual_overlay={"opacity": 0.1, "position_tolerance": 5}
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_hidden_texts(self):
        """Test hidden text generation."""
        integrity_shield = IntegrityShield(self.config)
        
        target_answers = ["Option A", "True", "Sample explanation"]
        hidden_texts = integrity_shield._generate_hidden_texts(target_answers)
        
        assert len(hidden_texts) == 3
        assert "The correct answer is Option A" in hidden_texts[0]["text"]
        assert "The correct answer is True" in hidden_texts[1]["text"]
        assert "The correct answer is Sample explanation" in hidden_texts[2]["text"]
    
    def test_generate_perturbation_signatures(self):
        """Test perturbation signature generation."""
        integrity_shield = IntegrityShield(self.config)
        
        document_data = {
            "id": "test_doc",
            "questions": [
                {"correct_answer": "Option A"},
                {"correct_answer": "True"},
                {"correct_answer": "Sample explanation"}
            ]
        }
        
        signatures = integrity_shield.generate_perturbation_signatures(document_data)
        
        assert signatures["document_id"] == "test_doc"
        assert "hidden_text" in signatures["perturbation_types"]
        assert len(signatures["target_answers"]) == 3
        assert "Option A" in signatures["target_answers"]
        assert "True" in signatures["target_answers"]
        assert "Sample explanation" in signatures["target_answers"]
    
    def test_apply_hidden_text_perturbation(self):
        """Test hidden text perturbation application."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%dummy content")
        
        integrity_shield = IntegrityShield(self.config)
        
        target_answers = ["Option A", "True"]
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            result_file = integrity_shield.apply_hidden_text_perturbation(pdf_file, target_answers)
            
            assert result_file.exists()
            assert "hidden_text" in result_file.name
    
    def test_apply_font_remapping_perturbation(self):
        """Test font remapping perturbation application."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%dummy content")
        
        integrity_shield = IntegrityShield(self.config)
        
        remapping_rules = {"CNN": "RNN", "What is": "What is NOT"}
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            result_file = integrity_shield.apply_font_remapping_perturbation(pdf_file, remapping_rules)
            
            assert result_file.exists()
            assert "font_remap" in result_file.name
    
    def test_apply_visual_overlay_perturbation(self):
        """Test visual overlay perturbation application."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%dummy content")
        
        integrity_shield = IntegrityShield(self.config)
        
        overlay_content = ["Overlay 1: Option A", "Overlay 2: True"]
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            result_file = integrity_shield.apply_visual_overlay_perturbation(pdf_file, overlay_content)
            
            assert result_file.exists()
            assert "visual_overlay" in result_file.name


class TestPDFGenerationIntegration:
    """Integration tests for PDF generation pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'latex_dir': self.temp_dir,
            'pdf_dir': self.temp_dir,
            'perturbed_dir': self.temp_dir
        }
        self.config.get_latex_config.return_value = Mock(
            base_template="templates/base_template.tex",
            output_dir=self.temp_dir,
            pdf_output_dir=self.temp_dir,
            school_name="Test School",
            course_number="TEST-101",
            subject="Test Subject",
            term="2023"
        )
        self.config.get_integrity_shield_config.return_value = Mock(
            enabled=True,
            perturbation_types=["hidden_text"],
            hidden_text={"color": "white", "position_offset": [2, 3]},
            font_remapping={"unicode_mappings": {}},
            visual_overlay={"opacity": 0.1, "position_tolerance": 5}
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline(self):
        """Test complete PDF generation pipeline."""
        # Sample document data
        document_data = {
            "id": "test_doc",
            "title": "Test Assessment",
            "total_marks": 20,
            "questions": [
                {
                    "id": "q1",
                    "type": "mcq",
                    "text": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "correct_answer": "4",
                    "marks": 2
                },
                {
                    "id": "q2",
                    "type": "tf",
                    "text": "2+2 equals 4.",
                    "correct_answer": "True",
                    "marks": 2
                }
            ]
        }
        
        # Test LaTeX generation
        templates = LaTeXTemplates(self.config)
        latex_content = templates.generate_document_latex(document_data)
        
        assert "\\documentclass[12pt]{article}" in latex_content
        assert "What is 2+2?" in latex_content
        assert "2+2 equals 4." in latex_content
        
        # Test gold labels generation
        gold_labels = templates.generate_gold_labels(document_data)
        
        assert gold_labels["document_id"] == "test_doc"
        assert len(gold_labels["answers"]) == 2
        
        # Test IntegrityShield signatures
        integrity_shield = IntegrityShield(self.config)
        signatures = integrity_shield.generate_perturbation_signatures(document_data)
        
        assert signatures["document_id"] == "test_doc"
        assert "4" in signatures["target_answers"]
        assert "True" in signatures["target_answers"]


# Test fixtures
@pytest.fixture
def sample_latex_content():
    """Sample LaTeX content for testing."""
    return r"""
\documentclass[12pt]{article}
\usepackage{enumitem}
\begin{document}
\section*{Test Section}
\begin{enumerate}
\item What is 2+2?
\begin{enumerate}[label=(\alph*)]
    \item 3 \qquad \item 4 \qquad \item 5 \qquad \item 6
\end{enumerate}
\end{enumerate}
\end{document}
"""


@pytest.fixture
def sample_pdf_file():
    """Sample PDF file for testing."""
    return b"%PDF-1.4\n%dummy PDF content"


if __name__ == "__main__":
    pytest.main([__file__])
