#!/usr/bin/env python3
"""
Run the synthetic conversation evaluation tests and generate a report.

This script runs the integration test that makes actual API calls to OpenAI
and generates a comprehensive HTML report of the results.
"""

import os
import sys
import json
import subprocess
import webbrowser
import argparse
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run synthetic conversation evaluation tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with maximum verbosity")
    return parser.parse_args()

def run_tests(verbose=False, debug=False):
    """Run the integration tests and capture the output."""
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("Running integration test with actual API calls...")
    print("This may take several minutes as it makes multiple API calls to OpenAI.")
    print("Progress will be displayed as the test runs.")
    
    # Create test_results directory if it doesn't exist
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    
    # Build the command with appropriate verbosity
    cmd = ["pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if debug:
        cmd.append("-vs")  # -s shows print statements from the test
        
    cmd.extend([
        "test_synthetic_conversation.py::test_full_pipeline_with_range", 
        "-m", "integration"
    ])
    
    # Add environment variables for the test
    env = os.environ.copy()
    if verbose:
        env["VERBOSE"] = "1"
    if debug:
        env["DEBUG"] = "1"
    
    # Run the integration test
    try:
        print(f"Running command: {' '.join(cmd)}")
        
        # Use subprocess.Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Get the return code
        process.wait()
        
        # Check if there was any error output
        stderr = process.stderr.read()
        if stderr:
            print(stderr)
        
        # Check if the test passed
        return process.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def generate_html_report():
    """Generate an HTML report from the test results."""
    test_output_dir = "test_results"
    
    # Check if the test results directory exists
    if not os.path.exists(test_output_dir):
        print(f"Error: Test results directory '{test_output_dir}' not found.")
        return None
    
    # Find all conversation files
    conversation_files = [f for f in os.listdir(test_output_dir) if f.startswith("conversation_") and f.endswith(".json")]
    
    if not conversation_files:
        print(f"Error: No conversation files found in '{test_output_dir}'.")
        return None
    
    print(f"Generating HTML report from {len(conversation_files)} conversation files...")
    
    # Generate HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Synthetic Conversation Evaluation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3, h4, h5 {
                color: #2c3e50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 5px solid #007bff;
            }
            .conversation {
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .conversation-header {
                background-color: #f8f9fa;
                padding: 10px;
                margin-bottom: 15px;
                border-radius: 5px;
            }
            .exchange {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }
            .ai-message, .user-message {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
            }
            .ai-message {
                background-color: #f1f0f0;
            }
            .user-message {
                background-color: #e3f2fd;
            }
            .stage-label {
                font-weight: bold;
                color: #6c757d;
                margin-bottom: 5px;
            }
            .evaluation {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
            .skill-scores {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 15px;
            }
            .skill-score {
                flex: 1;
                min-width: 200px;
            }
            .progress-bar {
                height: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
                margin-top: 5px;
            }
            .progress {
                height: 100%;
                background-color: #007bff;
                border-radius: 5px;
            }
            .badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                color: white;
                font-weight: bold;
            }
            .badge-Bronze {
                background-color: #cd7f32;
            }
            .badge-Silver {
                background-color: #c0c0c0;
            }
            .badge-Gold {
                background-color: #ffd700;
                color: #333;
            }
            .summary {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Synthetic Conversation Evaluation Report</h1>
                <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
    """
    
    # Add summary section
    html += """
            <div class="summary">
                <h2>Summary</h2>
                <p>This report contains evaluations of synthetic networking conversations at different skill levels.</p>
                <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                    <thead>
                        <tr style="background-color: #f2f2f2;">
                            <th>Conversation ID</th>
                            <th>Skill Level</th>
                            <th>Total Score</th>
                            <th>Badge Level</th>
                            <th>Critical Thinking</th>
                            <th>Communication</th>
                            <th>Emotional Intelligence</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Process each conversation file
    for file_name in sorted(conversation_files):
        file_path = os.path.join(test_output_dir, file_name)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversation = data.get("conversation", [])
        eval_data = data.get("evaluation", {})
        skill_scores = eval_data.get("skill_scores", {})
        badge_level = eval_data.get("badge_level", "Bronze")
        
        # Add to summary table
        html += f"""
                        <tr>
                            <td>{data.get("conversation_id", "Unknown")}</td>
                            <td>{data.get("skill_level", "Unknown")}</td>
                            <td>{eval_data.get("total_score", 0)}</td>
                            <td><span class="badge badge-{badge_level}">{badge_level}</span></td>
                            <td>{skill_scores.get("critical_thinking", 0)}</td>
                            <td>{skill_scores.get("communication", 0)}</td>
                            <td>{skill_scores.get("emotional_intelligence", 0)}</td>
                        </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
    """
    
    # Process each conversation file
    for file_name in sorted(conversation_files):
        file_path = os.path.join(test_output_dir, file_name)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversation = data.get("conversation", [])
        eval_data = data.get("evaluation", {})
        skill_scores = eval_data.get("skill_scores", {})
        badge_level = eval_data.get("badge_level", "Bronze")
        
        html += f"""
        <div class="conversation">
            <div class="conversation-header">
                <h3>Conversation: {data.get("conversation_id", "Unknown")}</h3>
                <p><strong>Skill Level:</strong> {data.get("skill_level", "Unknown")}</p>
                <p><strong>Persona:</strong> {data.get("persona", "Unknown")}</p>
            </div>
        """
        
        # Add conversation exchanges
        for exchange in conversation:
            stage = exchange.get("stage", "unknown")
            ai_prompt = exchange.get("ai_prompt", "")
            user_response = exchange.get("user_response", "")
            
            html += f"""
            <div class="exchange">
                <div class="stage-label">{stage.upper()}</div>
                <div class="ai-message">
                    <strong>AI:</strong> {ai_prompt}
                </div>
                <div class="user-message">
                    <strong>User:</strong> {user_response}
                </div>
            </div>
            """
        
        # Add evaluation section
        html += f"""
            </div>
            
            <div class="evaluation">
                <h4>Evaluation</h4>
                <p><strong>Total Score:</strong> {eval_data.get("total_score", 0)}/15</p>
                <p><strong>Badge Level:</strong> <span class="badge badge-{badge_level}">{badge_level}</span></p>
                
                <div class="skill-scores">
        """
        
        # Add skill scores
        for skill, score in skill_scores.items():
            max_score = 5  # Assuming max score is 5
            percentage = (score / max_score) * 100
            
            html += f"""
                    <div class="skill-score">
                        <h5>{skill.replace('_', ' ').title()}</h5>
                        <p>{score}/{max_score}</p>
                        <div class="progress-bar">
                            <div class="progress" style="width: {percentage}%;"></div>
                        </div>
                    </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Write the HTML to a file
    report_path = os.path.join(test_output_dir, "detailed_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"Detailed HTML report generated: {report_path}")
    return report_path

def main():
    """Run tests and generate report."""
    # Parse command line arguments
    args = parse_args()
    
    # Run the tests
    if not run_tests(verbose=args.verbose, debug=args.debug):
        print("Tests failed. See output above for details.")
        return 1
    
    # Generate HTML report
    report_path = generate_html_report()
    if report_path:
        print(f"Opening report in browser...")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 