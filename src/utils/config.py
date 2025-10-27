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
    
    def is_hierarchical_system_enabled(self) -> bool:
        """Check if hierarchical system is enabled."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('use_hierarchical_system', False)
    
    def get_domains_to_generate(self) -> List[str]:
        """Get list of domains to generate papers for."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('domains_to_generate', [])
    
    def get_academic_levels(self) -> List[str]:
        """Get list of academic levels for hierarchical generation."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('academic_levels', ['K-12', 'Undergraduate', 'Graduate'])
    
    def get_academic_levels_for_domain(self, domain: str) -> List[str]:
        """Get academic levels for a specific domain."""
        domain_config = self.get_domain_generation_config()
        domain_academic_levels = domain_config.get('domain_academic_levels', {})
        return domain_academic_levels.get(domain, ['K-12', 'Undergraduate', 'Graduate'])
    
    def get_domains_and_levels(self) -> Dict[str, List[str]]:
        """Get all domains with their appropriate academic levels."""
        domain_config = self.get_domain_generation_config()
        domains_to_generate = domain_config.get('domains_to_generate', [])
        domain_academic_levels = domain_config.get('domain_academic_levels', {})
        
        result = {}
        for domain in domains_to_generate:
            result[domain] = domain_academic_levels.get(domain, ['K-12', 'Undergraduate', 'Graduate'])
        
        return result
    
    
    def get_papers_per_domain_level(self) -> int:
        """Get number of papers to generate per domain-level combination."""
        domain_config = self.get_domain_generation_config()
        return domain_config.get('papers_per_domain_level', 1)
    
    
    def get_hierarchical_combinations(self) -> Dict[str, List[List[str]]]:
        """Get hierarchical domain combinations."""
        return self.config['document_generation'].get('hierarchical_combinations', {})
    
    def get_hierarchical_combination(self, domain: str, level: str) -> List[List[str]]:
        """Get question combinations for a specific domain and academic level."""
        hierarchical_combinations = self.get_hierarchical_combinations()
        
        # Map level names to the format used in config
        level_mapping = {
            "K-12": "k12",
            "Undergraduate": "undergraduate", 
            "Graduate": "graduate"
        }
        
        level_key = level_mapping.get(level, level.lower())
        key = f"{domain}_{level_key}"
        return hierarchical_combinations.get(key, [])
    
    def get_hierarchical_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get all hierarchical subject mappings."""
        mappings = {}
        for key, value in self.config.get('datasets', {}).items():
            if isinstance(value, dict) and 'academic_level' in value:
                mappings[key] = value
        return mappings
    
    def get_hierarchical_mapping(self, domain: str, level: str) -> Dict[str, Any]:
        """Get hierarchical mapping for a specific domain and academic level."""
        # Map level names to abbreviated forms used in config
        level_mapping = {
            "K-12": "k12",
            "Undergraduate": "undergrad", 
            "Graduate": "grad"
        }
        
        # Map domain names to abbreviated forms used in config
        domain_mapping = {
            "mathematics": "math",
            "computer_science": "cs",
            "machine_learning": "machine_learning",  # Keep as is
            "political_science": "political_science",  # Keep as is
            "religious_studies": "religious_studies",  # Keep as is
            "cybersecurity": "cybersecurity"  # Keep as is
        }
        
        level_abbrev = level_mapping.get(level, level.lower())
        domain_abbrev = domain_mapping.get(domain, domain)
        key = f"mmlu_{domain_abbrev}_{level_abbrev}"
        
        mappings = self.get_hierarchical_mappings()
        return mappings.get(key, {})
    
    def get_subjects_for_domain_level(self, domain: str, level: str) -> List[str]:
        """Get subjects for a specific domain and academic level."""
        mapping = self.get_hierarchical_mapping(domain, level)
        return mapping.get('subjects', [])
    
    
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
