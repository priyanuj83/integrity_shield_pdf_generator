"""
Configuration management for IntegrityShield system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset sources."""
    url: str
    type: str
    format: str
    name: str


@dataclass
class DocumentConfig:
    """Configuration for document generation."""
    total_documents: int
    initial_batch: int
    marks_per_document: int
    combinations: list


@dataclass
class LaTeXConfig:
    """Configuration for LaTeX processing."""
    base_template: str
    output_dir: str
    pdf_output_dir: str
    school_name: str
    course_number: str
    subject: str
    term: str


@dataclass
class IntegrityShieldConfig:
    """Configuration for IntegrityShield perturbations."""
    enabled: bool
    perturbation_types: list
    hidden_text: dict
    font_remapping: dict
    visual_overlay: dict


class ConfigManager:
    """Manages configuration for the IntegrityShield system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure."""
        required_sections = ['datasets', 'document_generation', 'latex', 'integrity_shield', 'logging', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for specific dataset."""
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset configuration not found: {dataset_name}")
        
        dataset_config = self.config['datasets'][dataset_name]
        
        # Handle different dataset sources
        if dataset_config.get('source') == 'local':
            url = dataset_config.get('path', '')
        elif dataset_config.get('source') == 'huggingface':
            repo = dataset_config.get('repo', '')
            config = dataset_config.get('config', '')
            url = f"{repo}/{config}"
        else:
            url = dataset_config.get('url', '')
        
        return DatasetConfig(
            url=url,
            type=dataset_config['type'],
            format=dataset_config['format'],
            name=dataset_name
        )
    
    def get_document_config(self) -> DocumentConfig:
        """Get document generation configuration."""
        doc_config = self.config['document_generation']
        return DocumentConfig(
            total_documents=doc_config['total_documents'],
            initial_batch=doc_config['initial_batch'],
            marks_per_document=doc_config['marks_per_document'],
            combinations=doc_config['combinations']
        )
    
    def get_integrity_shield_config(self) -> IntegrityShieldConfig:
        """Get IntegrityShield perturbation configuration."""
        shield_config = self.config.get('integrity_shield', {})
        return IntegrityShieldConfig(
            enabled=shield_config.get('enabled', False),
            perturbation_types=shield_config.get('perturbation_types', []),
            hidden_text=shield_config.get('hidden_text', {}),
            font_remapping=shield_config.get('font_remapping', {}),
            visual_overlay=shield_config.get('visual_overlay', {})
        )

    def get_latex_config(self) -> LaTeXConfig:
        """Get LaTeX configuration."""
        latex_config = self.config['latex']
        return LaTeXConfig(
            base_template=latex_config['base_template'],
            output_dir=latex_config['output_dir'],
            pdf_output_dir=latex_config['pdf_output_dir'],
            school_name=latex_config['school_name'],
            course_number=latex_config['course_number'],
            subject=latex_config['subject'],
            term=latex_config['term']
        )
    
    def get_pdf_generation_config(self) -> Dict[str, Any]:
        """Get PDF generation configuration."""
        return self.config.get('pdf_generation', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']
    
    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directory configuration."""
        output_config = self.config.get('output', {})
        output_config.setdefault('perturbed_dir', 'output/perturbed_documents')
        return output_config
    
    def get_question_combinations(self) -> list:
        """Get question type combinations for document generation."""
        return self.config['document_generation']['combinations']
    
    def get_domain_generation_config(self) -> Dict[str, Any]:
        """Get domain generation configuration."""
        return self.config['document_generation'].get('domain_generation', {})
    
    def is_domain_generation_enabled(self) -> bool:
        """Check if domain-specific generation is enabled."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('enabled', False)
    
    def get_domains_to_generate(self) -> List[str]:
        """Get list of domains to generate papers for."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('domains_to_generate', [])
    
    def get_papers_per_domain(self) -> int:
        """Get number of papers to generate per domain."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('papers_per_domain', 8)
    
    def get_domain_combinations(self) -> Dict[str, List[List[str]]]:
        """Get domain-specific question combinations."""
        return self.config['document_generation'].get('domain_combinations', {})
    
    def get_domain_combination(self, domain: str) -> List[List[str]]:
        """Get question combinations for a specific domain."""
        domain_combinations = self.get_domain_combinations()
        return domain_combinations.get(domain, [])
    
    def get_mmlu_subjects_for_domain(self, domain: str) -> List[str]:
        """Get MMLU subjects for a specific domain."""
        # Map domain names to config keys
        domain_mapping = {
            'mathematics': 'mmlu_math_subjects',
            'physics': 'mmlu_physics_subjects',
            'chemistry': 'mmlu_chemistry_subjects',
            'biology': 'mmlu_biology_subjects',
            'astronomy': 'mmlu_astronomy_subjects',
            'computer_science_theory': 'mmlu_computer_science_theory_subjects',
            'cybersecurity': 'mmlu_cybersecurity_subjects',
            'ai_ml': 'mmlu_ai_ml_subjects',
            'psychology': 'mmlu_psychology_subjects',
            'economics': 'mmlu_economics_subjects',
            'history': 'mmlu_history_subjects',
            'geography': 'mmlu_geography_subjects',
            'political_science': 'mmlu_political_science_subjects',
            'sociology': 'mmlu_sociology_subjects',
            'philosophy': 'mmlu_philosophy_subjects',
            'religious_studies': 'mmlu_religious_studies_subjects',
            'medical_sciences': 'mmlu_medical_sciences_subjects',
            'legal_studies': 'mmlu_legal_studies_subjects',
            'business_management': 'mmlu_business_management_subjects'
        }
        
        domain_key = domain_mapping.get(domain, f"mmlu_{domain}_subjects")
        if domain_key in self.config.get('datasets', {}):
            return self.config['datasets'][domain_key].get('subjects', [])
        return []
    
    def is_pdf_validation_enabled(self) -> bool:
        """Check if PDF validation is enabled."""
        pdf_config = self.get_pdf_generation_config()
        return pdf_config.get('validation_enabled', True)


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: str = "config.yaml") -> ConfigManager:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


def reload_config(config_path: str = "config.yaml") -> ConfigManager:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = ConfigManager(config_path)
    return _config_instance
