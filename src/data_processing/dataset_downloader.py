"""
Dataset downloader for IntegrityShield system.
Downloads datasets from Hugging Face and processes existing data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datasets import load_dataset

from ..utils.logger import get_logger, log_data_processing, log_success, log_error
from ..utils.config import get_config


class DatasetDownloader:
    """Downloads and processes datasets for IntegrityShield."""
    
    def __init__(self, config_manager=None, force_download: bool = False):
        """Initialize dataset downloader."""
        self.config = config_manager or get_config()
        self.logger = get_logger()
        self.output_dir = Path(self.config.get_output_dirs()['raw_data_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force_download = force_download
    
    def download_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Download a specific dataset from Hugging Face or load existing data."""
        output_file = self.output_dir / f"{dataset_name}.json"

        if output_file.exists() and not self.force_download:
            self.logger.info(f"Using cached dataset: {dataset_name}")
            with open(output_file, 'r') as f:
                return json.load(f)

        try:
            self.logger.info(f"Loading dataset: {dataset_name}")

            if dataset_name == "gsm8k_mcq":
                # Use existing GSM MCQ data
                processed_data = self._load_existing_gsm_mcq()
            else:
                # Download from Hugging Face
                processed_data = self._download_huggingface_dataset(dataset_name)

            # Save to file
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)

            log_success(f"Dataset load: {dataset_name}", f"Saved to {output_file}")
            return processed_data

        except Exception as e:
            log_error(e, f"Loading dataset {dataset_name}")
            raise

    def _load_existing_gsm_mcq(self) -> Dict[str, Any]:
        """Load existing GSM MCQ data from local directory."""
        gsm_mcq_dir = Path("data/gsm_mcq")
        
        processed_data = {
            "type": "math",
            "total_items": 0,
            "items": []
        }
        
        # Load from various GSM MCQ files
        for file_path in gsm_mcq_dir.rglob("*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            # Handle GSM MCQ format with A, B, C, D options
                            options = []
                            if "A" in item and "B" in item and "C" in item and "D" in item:
                                options = [item["A"], item["B"], item["C"], item["D"]]
                            else:
                                options = item.get("options", [])
                            
                            processed_item = {
                                "id": item.get("id", len(processed_data["items"])),
                                "question": item.get("Question", item.get("question", "")),
                                "answer": item.get("Answer", item.get("answer", "")),
                                "options": options,
                                "grade": item.get("grade", "unknown"),
                                "source_file": str(file_path)
                            }
                            processed_data["items"].append(processed_item)
                            processed_data["total_items"] += 1
                            
            except Exception as e:
                self.logger.warning(f"Could not load {file_path}: {e}")
        
        self.logger.info(f"Loaded {processed_data['total_items']} items from existing GSM MCQ data")
        return processed_data
    
    def _download_huggingface_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Download dataset from Hugging Face."""
        dataset_configs = {
            "mbpp_plus": ("Muennighoff/mbpp", "full"),
            "mmlu_abstract_algebra": ("cais/mmlu", "abstract_algebra"),
            "mmlu_all": ("cais/mmlu", "all"),
            "mmlu_anatomy": ("cais/mmlu", "anatomy"),
            "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "default")
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        repo_name, config_name = dataset_configs[dataset_name]
        self.logger.info(f"Downloading from Hugging Face: {repo_name}/{config_name}")
        
        # Load dataset from Hugging Face
        dataset = load_dataset(repo_name, config_name)
        
        # Process based on dataset type
        if dataset_name.startswith("mbpp"):
            return self._process_mbpp_dataset(dataset)
        elif dataset_name.startswith("mmlu"):
            if dataset_name == "mmlu_pro":
                return self._process_mmlu_pro_dataset(dataset)
            else:
                return self._process_mmlu_dataset(dataset)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
    
    def _process_mbpp_dataset(self, dataset) -> Dict[str, Any]:
        """Process MBPP dataset from Hugging Face."""
        processed = {
            "type": "coding",
            "total_items": 0,
            "items": []
        }
        
        # Process train and test splits
        for split_name, split_data in dataset.items():
            for item in split_data:
                processed_item = {
                    "id": item.get("task_id", len(processed["items"])),
                    "text": item.get("text", ""),
                    "code": item.get("code", ""),
                    "test_list": item.get("test_list", []),
                    "challenge_test_list": item.get("challenge_test_list", []),
                    "difficulty": item.get("difficulty", "unknown"),
                    "split": split_name
                }
                processed["items"].append(processed_item)
                processed["total_items"] += 1
        
        return processed
    
    def _process_mmlu_dataset(self, dataset) -> Dict[str, Any]:
        """Process MMLU dataset from Hugging Face."""
        processed = {
            "type": "social_science",
            "total_items": 0,
            "items": []
        }
        
        # Process all splits
        for split_name, split_data in dataset.items():
            for item in split_data:
                processed_item = {
                    "id": item.get("id", len(processed["items"])),
                    "question": item.get("question", ""),
                    "choices": item.get("choices", []),
                    "answer": item.get("answer", ""),
                    "subject": item.get("subject", "unknown"),
                    "split": split_name
                }
                processed["items"].append(processed_item)
                processed["total_items"] += 1
        
        return processed
    
    def _process_mmlu_pro_dataset(self, dataset) -> Dict[str, Any]:
        """Process MMLU-Pro dataset from Hugging Face."""
        processed = {
            "type": "comprehensive",
            "total_items": 0,
            "items": []
        }
        
        # Process all splits
        for split_name, split_data in dataset.items():
            for item in split_data:
                processed_item = {
                    "id": item.get("question_id", len(processed["items"])),
                    "question": item.get("question", ""),
                    "choices": item.get("options", []),
                    "answer": item.get("answer", ""),
                    "answer_index": item.get("answer_index", 0),
                    "subject": item.get("category", "unknown"),
                    "cot_content": item.get("cot_content", ""),
                    "source": item.get("src", "unknown"),
                    "split": split_name
                }
                processed["items"].append(processed_item)
                processed["total_items"] += 1
        
        return processed
    
    def _process_coding_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process coding/MBPP+ data."""
        processed = {
            "type": "coding",
            "total_items": len(data),
            "items": []
        }
        
        for item in data:
            processed_item = {
                "id": item.get("task_id", len(processed["items"])),
                "text": item.get("text", ""),
                "code": item.get("code", ""),
                "test_list": item.get("test_list", []),
                "challenge_test_list": item.get("challenge_test_list", []),
                "difficulty": item.get("difficulty", "unknown")
            }
            processed["items"].append(processed_item)
        
        return processed
    
    def _process_math_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process math/GSM8K data."""
        processed = {
            "type": "math",
            "total_items": len(data),
            "items": []
        }
        
        for item in data:
            processed_item = {
                "id": item.get("question", "").split()[0] if item.get("question") else len(processed["items"]),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "grade": item.get("grade", "unknown")
            }
            processed["items"].append(processed_item)
        
        return processed
    
    def _process_social_science_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process social science/MMLU data."""
        processed = {
            "type": "social_science",
            "total_items": len(data),
            "items": []
        }
        
        for item in data:
            processed_item = {
                "id": item.get("id", len(processed["items"])),
                "question": item.get("question", ""),
                "choices": item.get("choices", []),
                "answer": item.get("answer", ""),
                "subject": item.get("subject", "unknown")
            }
            processed["items"].append(processed_item)
        
        return processed
    
    def _process_summarization_data(self, data: List[str]) -> Dict[str, Any]:
        """Process summarization/CNN-DailyMail data."""
        processed = {
            "type": "summarization",
            "total_items": len(data),
            "items": []
        }
        
        for i, url in enumerate(data):
            if url.strip():
                processed_item = {
                    "id": i,
                    "url": url.strip(),
                    "type": "summarization"
                }
                processed["items"].append(processed_item)
        
        return processed
    
    def download_all_datasets(self) -> Dict[str, Any]:
        """Download all configured datasets."""
        self.logger.info("Starting dataset loading phase")
        
        try:
            # Define available datasets
            available_datasets = [
                "gsm8k_mcq",  # Use existing data
                "mbpp_plus",  # Hugging Face
                "mmlu_abstract_algebra",  # Hugging Face
                "mmlu_all",  # Hugging Face
                "mmlu_anatomy",  # Hugging Face
                "mmlu_pro"  # Hugging Face - MMLU-Pro
            ]
            
            results = {}
            
            for dataset_name in available_datasets:
                try:
                    self.logger.info(f"Loading {dataset_name}...")
                    results[dataset_name] = self.download_dataset(dataset_name)
                    
                    # Add delay to be respectful to servers
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {dataset_name}: {e}")
                    results[dataset_name] = {"error": str(e)}
            
            # Log results
            successful_downloads = sum(1 for r in results.values() if "error" not in r)
            total_datasets = len(results)
            
            log_success(f"Dataset loading completed", f"{successful_downloads}/{total_datasets} datasets loaded successfully")
            
            return results
            
        except Exception as e:
            log_error(e, "Dataset loading phase")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a downloaded dataset."""
        dataset_file = self.output_dir / f"{dataset_name}.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        return {
            "name": dataset_name,
            "type": data.get("type", "unknown"),
            "total_items": data.get("total_items", 0),
            "file_path": str(dataset_file),
            "file_size": dataset_file.stat().st_size
        }
    
    def list_downloaded_datasets(self) -> List[Dict[str, Any]]:
        """List all downloaded datasets."""
        datasets = []
        
        for dataset_file in self.output_dir.glob("*.json"):
            try:
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                
                datasets.append({
                    "name": dataset_file.stem,
                    "type": data.get("type", "unknown"),
                    "total_items": data.get("total_items", 0),
                    "file_path": str(dataset_file),
                    "file_size": dataset_file.stat().st_size
                })
            except Exception as e:
                self.logger.error(f"Error reading dataset file {dataset_file}: {e}")
        
        return datasets


def main():
    """Main function for testing dataset downloader."""
    from ..utils.logger import setup_logging
    
    # Setup logging
    config = get_config()
    setup_logging(config.get_logging_config())
    
    # Initialize downloader
    downloader = DatasetDownloader()
    
    # Download all datasets
    results = downloader.download_all_datasets()
    
    # Print results
    print("\nDownload Results:")
    for dataset_name, result in results.items():
        if "error" in result:
            print(f"❌ {dataset_name}: {result['error']}")
        else:
            print(f"✅ {dataset_name}: {result['total_items']} items")


if __name__ == "__main__":
    main()
