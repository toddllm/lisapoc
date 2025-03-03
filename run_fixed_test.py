#!/usr/bin/env python3
"""
Run the test with the fixed assertion and debug output.
"""

import os
import sys
import subprocess
import time

def run_test_with_debug():
    """Run the test with debug output."""
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here")
        return 1
    
    print("Running test with fixed assertion and DEBUG output...")
    print("This will show maximum verbosity including conversation content.")
    
    # Run the test with debug flag
    cmd = ["python", "run_evaluation_tests.py", "--debug"]
    
    try:
        # Use subprocess.Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )
        
        # Stream output in real-time
        print("\n--- OUTPUT BEGINS ---\n")
        for line in process.stdout:
            print(line, end='')
        
        # Get the return code
        process.wait()
        
        # Check if there was any error output
        stderr = process.stderr.read()
        if stderr:
            print("\n--- ERROR OUTPUT ---\n")
            print(stderr)
        
        print("\n--- OUTPUT ENDS ---\n")
        
        # Check if the test passed
        if process.returncode == 0:
            print("\nTest passed successfully!")
            
            # Check for test results
            if os.path.exists("test_results"):
                print("\nTest results available in test_results directory.")
                print("You can view the detailed report by opening:")
                print("  test_results/detailed_report.html")
            
            return 0
        else:
            print("\nTest failed. See output above for details.")
            return 1
    
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_test_with_debug()) 