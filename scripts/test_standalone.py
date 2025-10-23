#!/usr/bin/env python3
"""
Test script for the standalone PDF generation system.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config import get_config
        from src.data_processing.dataset_downloader import DatasetDownloader
        from src.data_processing.question_generator import QuestionGenerator
        from src.pdf_generation.latex_templates import LaTeXTemplates
        from src.pdf_generation.pdf_compiler import PDFCompiler
        from main import PDFGenerationPipeline
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import get_config
        
        config = get_config()
        
        # Test key configuration sections
        required_sections = ['datasets', 'document_generation', 'latex', 'logging']
        missing_sections = [section for section in required_sections if section not in config.config]
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")
    
    try:
        from main import PDFGenerationPipeline
        
        pipeline = PDFGenerationPipeline()
        
        # Check that all components are initialized
        assert hasattr(pipeline, 'dataset_downloader')
        assert hasattr(pipeline, 'question_generator')
        assert hasattr(pipeline, 'latex_templates')
        assert hasattr(pipeline, 'pdf_compiler')
        assert hasattr(pipeline, 'output_dirs')
        
        print("✅ Pipeline initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src",
        "src/data_processing",
        "src/pdf_generation", 
        "src/testing",
        "src/utils",
        "scripts",
        "data",
        "output",
        "logs",
        "templates"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("STANDALONE PDF GENERATION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_pipeline_initialization,
        test_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✅ All tests passed! The standalone system is ready.")
        print("\nNext steps:")
        print("1. Run: python3 generate_pdfs.py")
        print("2. Or run: python3 main.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
