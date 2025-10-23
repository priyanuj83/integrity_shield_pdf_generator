#!/usr/bin/env python3
"""
Simple test script to verify the IntegrityShield system structure.
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test if all required directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        "src",
        "src/data_processing", 
        "src/pdf_generation",
        "src/testing",
        "src/utils",
        "scripts",
        "data",
        "data/raw",
        "data/processed", 
        "data/gold_labels",
        "output",
        "output/latex_documents",
        "output/pdf_documents",
        "output/perturbed_documents",
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

def test_file_structure():
    """Test if all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "config.yaml",
        "requirements.txt", 
        "main.py",
        "scripts/run_initial_batch.py",
        "scripts/generate_pdfs.py",
        "README.md",
        "src/__init__.py",
        "src/data_processing/__init__.py",
        "src/data_processing/dataset_downloader.py",
        "src/data_processing/question_generator.py",
        "src/pdf_generation/__init__.py",
        "src/pdf_generation/latex_templates.py",
        "src/pdf_generation/pdf_compiler.py",
        "src/pdf_generation/integrity_shield.py",
        "src/testing/__init__.py",
        "src/testing/test_data_processing.py",
        "src/testing/test_pdf_generation.py",
        "src/testing/test_integrity_shield.py",
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "src/utils/config.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_config_file():
    """Test if config file is valid YAML."""
    print("\nTesting configuration file...")
    
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['datasets', 'document_generation', 'latex', 'integrity_shield', 'logging', 'output']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        else:
            print("✅ Configuration file is valid")
            return True
            
    except ImportError:
        print("⚠️  PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return False

def test_python_syntax():
    """Test if Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "main.py",
        "scripts/run_initial_batch.py",
        "scripts/generate_pdfs.py",
        "src/__init__.py",
        "src/data_processing/__init__.py",
        "src/data_processing/dataset_downloader.py",
        "src/data_processing/question_generator.py",
        "src/pdf_generation/__init__.py", 
        "src/pdf_generation/latex_templates.py",
        "src/pdf_generation/pdf_compiler.py",
        "src/pdf_generation/integrity_shield.py",
        "src/testing/__init__.py",
        "src/testing/test_data_processing.py",
        "src/testing/test_pdf_generation.py",
        "src/testing/test_integrity_shield.py",
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "src/utils/config.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("INTEGRITYSHIELD SYSTEM STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_file_structure,
        test_config_file,
        test_python_syntax
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
        print("✅ All tests passed! System structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run initial batch: python3 run_initial_batch.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
