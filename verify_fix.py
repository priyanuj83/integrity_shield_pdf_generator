#!/usr/bin/env python3
"""Verify the fix works with the exact problematic examples from the LaTeX file."""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# Import the function
from main import clean_option_text

def verify_fix():
    """Verify the fix with the exact problematic examples."""
    
    # These are the problematic options we saw in the LaTeX file
    problematic_cases = [
        # From question 1: "over 216", "over 36", "over 9", "over 4"
        # These should have been "4/9 over 216", etc.
        ("4/9 over 216", "4/9 over 216", "Should preserve fraction at start"),
        ("4/9 over 36", "4/9 over 36", "Should preserve fraction at start"),
        ("4/9 over 9", "4/9 over 9", "Should preserve fraction at start"),
        ("4/9 over 4", "4/9 over 4", "Should preserve fraction at start"),
        
        # From question 2: ",000", ",800"
        # These should have been "48,000", "48,800"
        ("48,000", "48,000", "Should preserve number with comma"),
        ("48,800", "48,800", "Should preserve number with comma"),
        
        # From question 3: "^7", "^3", "^49"
        # These should have been "7^3", "3^7", "49^1" or similar
        ("7^3", "7^3", "Should preserve exponent notation"),
        ("3^7", "3^7", "Should preserve exponent notation"),
        ("49^1", "49^1", "Should preserve exponent notation"),
        ("49", "49", "Should preserve plain number"),
    ]
    
    print("Verifying fix with problematic examples from LaTeX file...")
    print("=" * 70)
    
    all_passed = True
    
    for input_text, expected, description in problematic_cases:
        result = clean_option_text(input_text)
        if result == expected:
            print(f"[PASS] {description}")
            print(f"  '{input_text}' -> '{result}'")
        else:
            print(f"[FAIL] {description}")
            print(f"  Input:    '{input_text}'")
            print(f"  Output:   '{result}'")
            print(f"  Expected: '{expected}'")
            all_passed = False
        print()
    
    print("=" * 70)
    if all_passed:
        print("[SUCCESS] All problematic cases are now fixed!")
        print("\nNote: You need to regenerate the math papers to see the fix in the PDFs.")
        print("Run: python scripts/generate_single_domain.py mathematics --skip-download")
        return 0
    else:
        print("[ERROR] Some cases still fail!")
        return 1

if __name__ == "__main__":
    sys.exit(verify_fix())

