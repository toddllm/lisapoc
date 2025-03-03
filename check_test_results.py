#!/usr/bin/env python3
"""
Check for test results and display them.
"""

import os
import json
import glob
import pandas as pd

def check_test_results():
    """Check for test results and display them."""
    print("Checking for test results...")
    
    # Check if test_results directory exists
    if not os.path.exists("test_results"):
        print("No test_results directory found.")
        return False
    
    # Check for conversation files
    conversation_files = glob.glob("test_results/conversation_*.json")
    if not conversation_files:
        print("No conversation files found in test_results directory.")
        return False
    
    print(f"\nFound {len(conversation_files)} conversation files:")
    for file in conversation_files:
        print(f"  - {os.path.basename(file)}")
    
    # Check for summary CSV
    summary_file = "test_results/test_results_summary.csv"
    if os.path.exists(summary_file):
        print(f"\nFound summary file: {summary_file}")
        try:
            df = pd.read_csv(summary_file)
            print("\nSummary data:")
            print(df)
        except Exception as e:
            print(f"Error reading summary file: {e}")
    else:
        print(f"\nNo summary file found: {summary_file}")
    
    # Check for HTML report
    report_file = "test_results/detailed_report.html"
    if os.path.exists(report_file):
        print(f"\nFound HTML report: {report_file}")
        print(f"You can open it with: open {report_file}")
    else:
        print(f"\nNo HTML report found: {report_file}")
    
    # Display a sample conversation
    if conversation_files:
        print("\nDisplaying sample from first conversation file:")
        try:
            with open(conversation_files[0], 'r') as f:
                data = json.load(f)
            
            conversation = data.get("conversation", [])
            evaluation = data.get("evaluation", {})
            
            print(f"Conversation ID: {data.get('conversation_id', 'Unknown')}")
            print(f"Skill Level: {data.get('skill_level', 'Unknown')}")
            print(f"Total Score: {evaluation.get('total_score', 0)}")
            print(f"Badge Level: {evaluation.get('badge_level', 'Unknown')}")
            
            if conversation:
                print("\nSample exchanges:")
                for i, exchange in enumerate(conversation[:2]):  # Show first 2 exchanges
                    print(f"\nExchange {i+1} - Stage: {exchange.get('stage', 'unknown').upper()}")
                    print(f"AI: {exchange.get('ai_prompt', '')}")
                    print(f"User: {exchange.get('user_response', '')}")
                
                if len(conversation) > 2:
                    print("\n... (more exchanges available) ...")
        except Exception as e:
            print(f"Error reading conversation file: {e}")
    
    return True

if __name__ == "__main__":
    check_test_results() 