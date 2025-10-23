#!/usr/bin/env python3
"""
Test script for the updated dataset loading system.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

def test_gsm_mcq_loading():
    """Test loading existing GSM MCQ data."""
    print("Testing GSM MCQ data loading...")
    
    try:
        from src.data_processing.dataset_downloader import DatasetDownloader
        from src.utils.config import get_config
        
        # Initialize downloader
        config = get_config()
        downloader = DatasetDownloader(config)
        
        # Test loading GSM MCQ data
        print("Loading GSM MCQ data from local directory...")
        gsm_data = downloader.download_dataset("gsm8k_mcq")
        
        print(f"✅ Successfully loaded GSM MCQ data:")
        print(f"   - Type: {gsm_data['type']}")
        print(f"   - Total items: {gsm_data['total_items']}")
        print(f"   - Sample items: {len(gsm_data['items'][:3])}")
        
        if gsm_data['items']:
            sample_item = gsm_data['items'][0]
            print(f"   - Sample question: {sample_item.get('question', 'N/A')[:100]}...")
            print(f"   - Sample answer: {sample_item.get('answer', 'N/A')}")
            print(f"   - Sample options: {sample_item.get('options', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading GSM MCQ data: {e}")
        return False

def test_huggingface_loading():
    """Test loading Hugging Face datasets."""
    print("\nTesting Hugging Face dataset loading...")
    
    try:
        from src.data_processing.dataset_downloader import DatasetDownloader
        from src.utils.config import get_config
        
        # Initialize downloader
        config = get_config()
        downloader = DatasetDownloader(config)
        
        # Test loading a small Hugging Face dataset
        print("Loading MMLU abstract algebra dataset...")
        mmlu_data = downloader.download_dataset("mmlu_abstract_algebra")
        
        print(f"✅ Successfully loaded MMLU data:")
        print(f"   - Type: {mmlu_data['type']}")
        print(f"   - Total items: {mmlu_data['total_items']}")
        
        if mmlu_data['items']:
            sample_item = mmlu_data['items'][0]
            print(f"   - Sample question: {sample_item.get('question', 'N/A')[:100]}...")
            print(f"   - Sample choices: {sample_item.get('choices', [])}")
            print(f"   - Sample answer: {sample_item.get('answer', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading Hugging Face dataset: {e}")
        print("   This might be due to network issues or missing dependencies.")
        return False

def test_all_datasets():
    """Test loading all configured datasets."""
    print("\nTesting all dataset loading...")
    
    try:
        from src.data_processing.dataset_downloader import DatasetDownloader
        from src.utils.config import get_config
        
        # Initialize downloader
        config = get_config()
        downloader = DatasetDownloader(config)
        
        # Test loading all datasets
        print("Loading all configured datasets...")
        results = downloader.download_all_datasets()
        
        print(f"✅ Dataset loading results:")
        for dataset_name, result in results.items():
            if "error" in result:
                print(f"   ❌ {dataset_name}: {result['error']}")
            else:
                print(f"   ✅ {dataset_name}: {result.get('total_items', 0)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in dataset loading: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DATASET LOADING SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_gsm_mcq_loading,
        test_huggingface_loading,
        test_all_datasets
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
        print("✅ All dataset loading tests passed!")
        print("\nThe updated system is working correctly with:")
        print("- Local GSM MCQ data loading")
        print("- Hugging Face dataset integration")
        print("- Multiple dataset support")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
