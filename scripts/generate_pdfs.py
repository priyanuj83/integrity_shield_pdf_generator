#!/usr/bin/env python3
"""Convenience wrapper around the IntegrityShield Gen2 generator."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IntegrityShield PDFs")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--count", type=int, default=5, help="Number of documents to generate")
    parser.add_argument("--skip-download", action="store_true", help="Reuse cached datasets")
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def build_main_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--config", args.config, "--count", str(args.count), "--seed", str(args.seed)]
    if args.skip_download:
        argv.append("--skip-download")
    if args.refresh_data:
        argv.append("--refresh-data")
    if args.verbose:
        argv.append("--verbose")
    return argv


def main() -> int:
    args = parse_args()
    from main import main as run_main  # import after adjusting sys.path
    return run_main(build_main_argv(args))


if __name__ == "__main__":
    sys.exit(main())
