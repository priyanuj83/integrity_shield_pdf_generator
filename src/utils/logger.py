"""
Logging utilities for IntegrityShield system.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class IntegrityShieldLogger:
    """Centralized logging system for IntegrityShield."""
    
    def __init__(self, config: dict):
        """Initialize logger with configuration."""
        self.config = config
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure loguru logger with settings from config."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        logger.add(
            sys.stdout,
            level=self.config.get('level', 'INFO'),
            format=self.config.get('format', 
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
            colorize=True
        )
        
        # File handler
        log_file = self.config.get('file', 'logs/integrity_shield.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=self.config.get('level', 'INFO'),
            format=self.config.get('format', 
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
            rotation=self.config.get('rotation', '10 MB'),
            retention=self.config.get('retention', '30 days'),
            compression="zip"
        )
    
    def get_logger(self):
        """Get configured logger instance."""
        return logger


def setup_logging(config: dict) -> IntegrityShieldLogger:
    """Setup and return configured logger."""
    return IntegrityShieldLogger(config)


def log_function_call(func_name: str, args: dict = None, result: any = None):
    """Log function call with arguments and result."""
    logger.info(f"Function call: {func_name}")
    if args:
        logger.debug(f"Arguments: {args}")
    if result is not None:
        logger.debug(f"Result: {result}")


def log_error(error: Exception, context: str = ""):
    """Log error with context."""
    logger.error(f"Error in {context}: {str(error)}")
    logger.exception("Full traceback:")


def log_success(operation: str, details: str = ""):
    """Log successful operation."""
    logger.success(f"Successfully completed: {operation}")
    if details:
        logger.info(f"Details: {details}")


def log_data_processing(stage: str, count: int, total: int = None):
    """Log data processing progress."""
    if total:
        percentage = (count / total) * 100
        logger.info(f"Data processing - {stage}: {count}/{total} ({percentage:.1f}%)")
    else:
        logger.info(f"Data processing - {stage}: {count} items processed")


def log_document_generation(doc_type: str, doc_id: str, status: str):
    """Log document generation progress."""
    logger.info(f"Document generation - {doc_type} (ID: {doc_id}): {status}")


def log_integrity_shield(operation: str, doc_id: str, perturbation_type: str = None):
    """Log IntegrityShield operations."""
    if perturbation_type:
        logger.info(f"IntegrityShield - {operation} on {doc_id} using {perturbation_type}")
    else:
        logger.info(f"IntegrityShield - {operation} on {doc_id}")


# Global logger instance
_logger_instance: Optional[IntegrityShieldLogger] = None


def get_logger():
    """Get global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        # Default configuration if not initialized
        default_config = {
            'level': 'INFO',
            'format': "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            'file': 'logs/integrity_shield.log',
            'rotation': '10 MB',
            'retention': '30 days'
        }
        _logger_instance = IntegrityShieldLogger(default_config)
    return _logger_instance.get_logger()
