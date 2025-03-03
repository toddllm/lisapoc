#!/usr/bin/env python3
"""
Verify if the verbose and debug code was properly added to the test files.
"""

import os
import re

def check_file_for_patterns(file_path, patterns):
    """Check if a file contains specific patterns."""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, content)
            results[name] = bool(match)
        
        return True, results
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    """Check test files for verbose and debug code."""
    print("Verifying changes to test files...")
    
    # Patterns to look for in run_evaluation_tests.py
    run_tests_patterns = {
        "verbose_arg": r"parser\.add_argument\(\"-v\", \"--verbose\"",
        "debug_arg": r"parser\.add_argument\(\"--debug\"",
        "run_tests_verbose": r"def run_tests\(verbose=False, debug=False\)",
        "env_verbose": r"if verbose:\s+env\[\"VERBOSE\"\] = \"1\"",
        "env_debug": r"if debug:\s+env\[\"DEBUG\"\] = \"1\""
    }
    
    # Patterns to look for in test_synthetic_conversation.py
    test_patterns = {
        "verbose_check": r"verbose = os\.environ\.get\(\"VERBOSE\"\) == \"1\"",
        "debug_check": r"debug = os\.environ\.get\(\"DEBUG\"\) == \"1\"",
        "verbose_print": r"if verbose or debug:",
        "timing_code": r"start_time = datetime\.now\(\)",
        "progress_indicators": r"print\(f\"\n\{\'-\'\*80\}\"\)"
    }
    
    # Check run_evaluation_tests.py
    success, results = check_file_for_patterns("run_evaluation_tests.py", run_tests_patterns)
    if success:
        print("\nrun_evaluation_tests.py:")
        for name, found in results.items():
            status = "✅ Found" if found else "❌ Not found"
            print(f"  {name}: {status}")
    else:
        print(f"\nError checking run_evaluation_tests.py: {results}")
    
    # Check test_synthetic_conversation.py
    success, results = check_file_for_patterns("test_synthetic_conversation.py", test_patterns)
    if success:
        print("\ntest_synthetic_conversation.py:")
        for name, found in results.items():
            status = "✅ Found" if found else "❌ Not found"
            print(f"  {name}: {status}")
    else:
        print(f"\nError checking test_synthetic_conversation.py: {results}")
    
    print("\nTo run tests with verbose output:")
    print("  python run_evaluation_tests.py --verbose")
    print("\nTo run tests with maximum verbosity (debug mode):")
    print("  python run_evaluation_tests.py --debug")

if __name__ == "__main__":
    main() 