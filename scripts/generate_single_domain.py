#!/usr/bin/env python3
"""
Script to generate question papers for a single domain.

Usage:
    python scripts/generate_single_domain.py history
    python scripts/generate_single_domain.py mathematics --skip-download --verbose
    python scripts/generate_single_domain.py physics --seed 123
"""

import sys
import argparse
import tempfile
import shutil
import traceback
from pathlib import Path

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml
from main import generate_documents, parse_args
from src.utils.config import get_config
from src.utils.logger import setup_logging


def create_temp_config(domain: str, original_config_path: Path) -> Path:
    """
    Create a temporary config file with only the specified domain.
    
    Args:
        domain: The domain name to generate papers for
        original_config_path: Path to the original config.yaml
        
    Returns:
        Path to the temporary config file
    """
    # Load original config
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Modify domains_to_generate to only include the specified domain
    if 'document_generation' in config and 'domain_generation' in config['document_generation']:
        config['document_generation']['domain_generation']['domains_to_generate'] = [domain]
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        prefix=f'config_{domain}_',
        delete=False,
        encoding='utf-8'
    )
    temp_config_path = Path(temp_config.name)
    
    # Write modified config
    yaml.dump(config, temp_config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    temp_config.close()
    
    return temp_config_path


def validate_domain(domain: str, config_path: Path) -> bool:
    """
    Validate that the domain exists in the config.
    
    Args:
        domain: The domain name to validate
        config_path: Path to the config file
        
    Returns:
        True if domain is valid, False otherwise
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'document_generation' in config and 'domain_generation' in config['document_generation']:
        valid_domains = config['document_generation']['domain_generation'].get('domains_to_generate', [])
        if domain in valid_domains:
            return True
    
    return False


def get_domain_info(domain: str, config_path: Path) -> dict:
    """
    Get information about the domain (academic levels, expected papers).
    
    Args:
        domain: The domain name
        config_path: Path to the config file
        
    Returns:
        Dictionary with domain information
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    domain_academic_levels = config.get('document_generation', {}).get('domain_generation', {}).get('domain_academic_levels', {})
    papers_per_domain_level = config.get('document_generation', {}).get('domain_generation', {}).get('papers_per_domain_level', 5)
    
    levels = domain_academic_levels.get(domain, ['K-12', 'Undergraduate', 'Graduate'])
    
    return {
        'domain': domain,
        'academic_levels': levels,
        'papers_per_level': papers_per_domain_level,
        'total_papers': len(levels) * papers_per_domain_level
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate question papers for a single domain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_single_domain.py history
  python scripts/generate_single_domain.py mathematics --skip-download --verbose
  python scripts/generate_single_domain.py physics --seed 123 --skip-download
  
Valid domains:
  science, mathematics, physics, chemistry, biology, computer_science,
  engineering, economics, psychology, philosophy, health, business,
  history, geography, sociology, political_science, religious_studies,
  machine_learning, cybersecurity, astronomy
        """
    )
    
    parser.add_argument(
        'domain',
        type=str,
        help='Domain name to generate papers for (e.g., history, mathematics, physics)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the original config file (default: config.yaml)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Reuse cached datasets (skip download)'
    )
    parser.add_argument(
        '--refresh-data',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--keep-temp-config',
        action='store_true',
        help='Keep the temporary config file after generation (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Validate domain
    if not validate_domain(args.domain, config_path):
        print(f"Error: Domain '{args.domain}' is not valid or not found in config.yaml")
        print(f"\nValid domains are:")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        valid_domains = config.get('document_generation', {}).get('domain_generation', {}).get('domains_to_generate', [])
        for d in valid_domains:
            print(f"  - {d}")
        sys.exit(1)
    
    # Get domain info
    domain_info = get_domain_info(args.domain, config_path)
    
    print("=" * 80)
    print(f"Generating papers for domain: {args.domain}")
    print(f"  Academic Levels: {', '.join(domain_info['academic_levels'])}")
    print(f"  Papers per level: {domain_info['papers_per_level']}")
    print(f"  Total papers to generate: {domain_info['total_papers']}")
    print("=" * 80)
    
    # Create temporary config
    temp_config_path = None
    try:
            temp_config_path = create_temp_config(args.domain, config_path)
            print(f"Created temporary config: {temp_config_path}")
            
            # Create argparse.Namespace object with the arguments
            gen_args = argparse.Namespace(
                config=str(temp_config_path),
                count=5,  # Not used in hierarchical mode, but required by parse_args
                skip_download=args.skip_download,
                refresh_data=args.refresh_data,
                seed=args.seed,
                verbose=args.verbose,
                max_compile_jobs=4,  # Default value for compilation jobs
                workers=6  # Default value for parallel workers
            )
            
            # Setup logging
            config_obj = get_config(str(temp_config_path))
            logging_config = config_obj.get_logging_config()
            if args.verbose:
                logging_config = dict(logging_config)
                logging_config["level"] = "DEBUG"
            setup_logging(logging_config)
            
            # Run the generation
            generate_documents(gen_args)
            
            print("\n" + "=" * 80)
            print(f"✓ Successfully generated {domain_info['total_papers']} papers for {args.domain}")
            print(f"  Output location: output/{args.domain}/")
            print("=" * 80)
            
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error during generation: {e}")
        traceback.print_exc()
        sys.exit(1)
    except SystemExit as e:
        # Re-raise SystemExit to preserve exit codes
        raise
    finally:
        # Clean up temporary config file
        if temp_config_path and not args.keep_temp_config:
            try:
                temp_config_path.unlink()
                print(f"\nCleaned up temporary config file")
            except Exception as e:
                print(f"Warning: Could not delete temporary config file: {e}")
                print(f"  You can manually delete: {temp_config_path}")


if __name__ == "__main__":
    main()

