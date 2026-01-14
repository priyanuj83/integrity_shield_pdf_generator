#!/usr/bin/env python3
"""Quick sanity check for clean_option_text function."""

import re
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# Import the function
from main import clean_option_text

def test_clean_option_text():
    """Test clean_option_text with various inputs."""
    
    test_cases = [
        # List markers that SHOULD be removed
        ("4. Answer text", "Answer text", "List marker with period"),
        ("(4) Answer text", "Answer text", "List marker with parens"),
        ("[4] Answer text", "Answer text", "List marker with brackets"),
        ("  4. Answer text", "Answer text", "List marker with leading space"),
        ("1. True", "True", "Simple list marker"),
        
        # Math content that SHOULD be preserved
        ("4/9 over 216", "4/9 over 216", "Fraction at start"),
        ("48,000", "48,000", "Number with comma"),
        ("7^3", "7^3", "Exponent notation"),
        ("4.5", "4.5", "Decimal number"),
        ("123", "123", "Plain number"),
        ("4/9", "4/9", "Simple fraction"),
        ("48,800", "48,800", "Number with comma"),
        ("7^7", "7^7", "Exponent"),
        ("3^7", "3^7", "Exponent"),
        ("49", "49", "Plain number"),
        
        # Edge cases
        ("", "", "Empty string"),
        ("   ", "", "Whitespace only"),
        ("No number here", "No number here", "No numbers at all"),
        ("Answer 4. text", "Answer 4. text", "Number not at start"),
    ]
    
    print("Testing clean_option_text function...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for input_text, expected, description in test_cases:
        result = clean_option_text(input_text)
        if result == expected:
            print(f"[PASS] {description}")
            print(f"  Input:    '{input_text}'")
            print(f"  Output:   '{result}'")
            print(f"  Expected: '{expected}'")
            passed += 1
        else:
            print(f"[FAIL] {description}")
            print(f"  Input:    '{input_text}'")
            print(f"  Output:   '{result}'")
            print(f"  Expected: '{expected}'")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n[ERROR] Some tests failed!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    test_clean_option_text()

