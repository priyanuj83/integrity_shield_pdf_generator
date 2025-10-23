"""
Tests for data processing components.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ..data_processing.dataset_downloader import DatasetDownloader
from ..data_processing.question_generator import QuestionGenerator, QuestionType
from ..utils.config import get_config


class TestDatasetDownloader:
    """Test cases for DatasetDownloader."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {'raw_data_dir': self.temp_dir}
        self.config.get_dataset_config.return_value = Mock(
            url="https://example.com/test.json",
            type="test",
            format="json",
            name="test_dataset"
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_download_dataset_success(self, mock_get):
        """Test successful dataset download."""
        # Mock response
        mock_response = Mock()
        mock_response.text = '{"test": "data"}'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Initialize downloader
        downloader = DatasetDownloader(self.config)
        
        # Test download
        result = downloader.download_dataset("test_dataset")
        
        # Assertions
        assert result["type"] == "test"
        assert result["total_items"] == 1
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_download_dataset_failure(self, mock_get):
        """Test dataset download failure."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        # Initialize downloader
        downloader = DatasetDownloader(self.config)
        
        # Test download failure
        with pytest.raises(Exception):
            downloader.download_dataset("test_dataset")
    
    def test_parse_jsonl(self):
        """Test JSONL parsing."""
        downloader = DatasetDownloader(self.config)
        
        jsonl_text = '{"item": 1}\n{"item": 2}\n'
        result = downloader._parse_jsonl(jsonl_text)
        
        assert len(result) == 2
        assert result[0]["item"] == 1
        assert result[1]["item"] == 2
    
    def test_process_coding_data(self):
        """Test coding data processing."""
        downloader = DatasetDownloader(self.config)
        
        test_data = [
            {"task_id": 1, "text": "Test task", "code": "print('hello')", "test_list": ["test1"]},
            {"task_id": 2, "text": "Another task", "code": "print('world')", "test_list": ["test2"]}
        ]
        
        result = downloader._process_coding_data(test_data)
        
        assert result["type"] == "coding"
        assert result["total_items"] == 2
        assert len(result["items"]) == 2
        assert result["items"][0]["id"] == 1
        assert result["items"][0]["text"] == "Test task"


class TestQuestionGenerator:
    """Test cases for QuestionGenerator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'raw_data_dir': self.temp_dir,
            'processed_data_dir': self.temp_dir
        }
        self.config.get_question_combinations.return_value = [
            ["mcq", "tf"],
            ["mcq", "long"],
            ["tf", "long"]
        ]
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_mcq_questions(self):
        """Test MCQ question generation."""
        # Create mock dataset
        dataset_data = {
            "type": "math",
            "total_items": 2,
            "items": [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"}
            ]
        }
        
        # Save mock dataset
        dataset_file = Path(self.temp_dir) / "gsm8k_mcq.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
        
        # Initialize generator
        generator = QuestionGenerator(self.config)
        
        # Test MCQ generation
        questions = generator.generate_mcq_questions(2, "math")
        
        assert len(questions) == 2
        assert questions[0].type == QuestionType.MCQ
        assert questions[0].subject == "math"
        assert questions[0].marks == 2
    
    def test_generate_true_false_questions(self):
        """Test True/False question generation."""
        generator = QuestionGenerator(self.config)
        
        questions = generator.generate_true_false_questions(3)
        
        assert len(questions) == 3
        assert all(q.type == QuestionType.TRUE_FALSE for q in questions)
        assert all(q.marks == 2 for q in questions)
    
    def test_generate_long_form_questions(self):
        """Test long-form question generation."""
        generator = QuestionGenerator(self.config)
        
        questions = generator.generate_long_form_questions(2)
        
        assert len(questions) == 2
        assert all(q.type == QuestionType.LONG_FORM for q in questions)
        assert all(q.marks == 5 for q in questions)
    
    def test_generate_document(self):
        """Test document generation."""
        generator = QuestionGenerator(self.config)
        
        document = generator.generate_document(
            "test_doc",
            [QuestionType.MCQ, QuestionType.TRUE_FALSE],
            20
        )
        
        assert document.id == "test_doc"
        assert document.total_marks <= 20
        assert len(document.question_types) == 2
        assert QuestionType.MCQ in document.question_types
        assert QuestionType.TRUE_FALSE in document.question_types
    
    def test_generate_initial_batch(self):
        """Test initial batch generation."""
        generator = QuestionGenerator(self.config)
        
        documents = generator.generate_initial_batch()
        
        assert len(documents) == 3  # Based on mock combinations
        assert all(doc.id.startswith("doc_") for doc in documents)
        assert all(doc.total_marks > 0 for doc in documents)


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.get_output_dirs.return_value = {
            'raw_data_dir': self.temp_dir,
            'processed_data_dir': self.temp_dir
        }
        self.config.get_question_combinations.return_value = [
            ["mcq", "tf"]
        ]
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_full_pipeline(self, mock_get):
        """Test complete data processing pipeline."""
        # Mock dataset download
        mock_response = Mock()
        mock_response.text = '{"question": "Test?", "answer": "4"}'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock config
        self.config.get_dataset_config.return_value = Mock(
            url="https://example.com/test.json",
            type="math",
            format="json",
            name="test_dataset"
        )
        
        # Test pipeline
        downloader = DatasetDownloader(self.config)
        generator = QuestionGenerator(self.config)
        
        # Download dataset
        dataset_data = downloader.download_dataset("test_dataset")
        
        # Generate questions
        questions = generator.generate_mcq_questions(1, "math")
        
        # Assertions
        assert dataset_data["type"] == "math"
        assert len(questions) == 1
        assert questions[0].type == QuestionType.MCQ


# Test fixtures
@pytest.fixture
def sample_question_data():
    """Sample question data for testing."""
    return {
        "id": "test_question",
        "type": "mcq",
        "text": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4",
        "marks": 2,
        "difficulty": "easy",
        "subject": "math"
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "id": "test_doc",
        "title": "Test Assessment",
        "total_marks": 20,
        "question_types": ["mcq", "tf"],
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


def test_question_validation(sample_question_data):
    """Test question data validation."""
    # Test required fields
    required_fields = ["id", "type", "text", "marks"]
    for field in required_fields:
        assert field in sample_question_data
    
    # Test question type
    assert sample_question_data["type"] in ["mcq", "tf", "long"]
    
    # Test marks
    assert sample_question_data["marks"] > 0


def test_document_validation(sample_document_data):
    """Test document data validation."""
    # Test required fields
    required_fields = ["id", "title", "total_marks", "questions"]
    for field in required_fields:
        assert field in sample_document_data
    
    # Test questions
    assert len(sample_document_data["questions"]) > 0
    
    # Test total marks calculation
    calculated_marks = sum(q["marks"] for q in sample_document_data["questions"])
    assert calculated_marks == sample_document_data["total_marks"]


if __name__ == "__main__":
    pytest.main([__file__])
