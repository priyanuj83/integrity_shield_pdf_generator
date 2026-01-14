#!/usr/bin/env python3
"""
Calculate question type distribution across all documents.

This script analyzes all JSON output files to determine the percentage
distribution of MCQ, True/False, and Long-Form questions across the dataset.
"""

import json
from pathlib import Path
from collections import Counter


def calculate_question_type_distribution():
    """Calculate the percentage distribution of MCQ, T/F, and Long-Form questions."""
    
    # Find all JSON files in output directories
    output_dir = Path(__file__).parent.parent / "output"
    json_files = list(output_dir.rglob("JSON_output/*.json"))
    
    print(f"Found {len(json_files)} documents")
    
    # Counter for question types
    totals = Counter()
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Use pre-computed statistics
            stats = data.get("question_statistics", {}).get("by_type", {})
            totals.update(stats)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Calculate total questions
    total = sum(totals.values())
    
    # Display results
    print("\n" + "=" * 60)
    print("QUESTION TYPE DISTRIBUTION")
    print("=" * 60)
    print(f"\nTotal questions across all documents: {total:,}")
    print(f"Total documents processed: {len(json_files)}")
    print("\n" + "-" * 60)
    print(f"{'Type':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 60)
    
    for qtype in ["MCQ", "TF", "LONG"]:
        count = totals[qtype]
        pct = count / total * 100 if total > 0 else 0
        label = "True/False" if qtype == "TF" else ("Long-Form" if qtype == "LONG" else "MCQ")
        print(f"{label:<15} {count:>10,} {pct:>11.2f}%")
    
    print("=" * 60)
    
    return {
        'total_questions': total,
        'total_documents': len(json_files),
        'counts': dict(totals),
        'percentages': {
            qtype: (totals[qtype] / total * 100) if total > 0 else 0
            for qtype in ['MCQ', 'TF', 'LONG']
        }
    }


if __name__ == "__main__":
    calculate_question_type_distribution()

