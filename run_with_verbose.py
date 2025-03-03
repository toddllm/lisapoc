#!/usr/bin/env python3
"""
Run the tests with verbose output and a timeout.
"""

import os
import sys
import subprocess
import time

def run_with_timeout(cmd, timeout=300):
    """Run a command with a timeout."""
    print(f"Running command: {' '.join(cmd)}")
    print(f"Timeout set to {timeout} seconds")
    
    start_time = time.time()
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )
        
        # Set up timeout
        elapsed = 0
        while process.poll() is None and elapsed < timeout:
            # Read and print any output
            for line in process.stdout:
                print(line, end='')
                break  # Just read one line at a time
            
            # Sleep a bit
            time.sleep(1)
            elapsed = time.time() - start_time
            
            # Print a progress message every 10 seconds
            if elapsed % 10 < 1:
                print(f"Still running... ({elapsed:.0f} seconds elapsed)")
        
        # Check if we timed out
        if elapsed >= timeout:
            print(f"\nTimeout after {timeout} seconds!")
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
            return False
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        
        return process.returncode == 0
    
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run the tests with verbose output."""
    print("Running tests with verbose output...")
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here")
        return 1
    
    # Run the single conversation generator first
    print("\n=== Running single conversation generator ===")
    success = run_with_timeout(["python", "generate_single_conversation.py"], timeout=120)
    
    if not success:
        print("Single conversation generator failed!")
        return 1
    
    # Run the tests with verbose output
    print("\n=== Running tests with verbose output ===")
    success = run_with_timeout(
        ["python", "run_evaluation_tests.py", "--verbose"], 
        timeout=600  # 10 minutes
    )
    
    if not success:
        print("Tests failed or timed out!")
        return 1
    
    print("Tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 