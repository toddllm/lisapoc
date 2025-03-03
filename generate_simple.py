#!/usr/bin/env python3
"""
Simple script to generate conversations and evaluations using direct OpenAI API calls.
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any
import httpx
from openai import OpenAI

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# System prompts
SYSTEM_PROMPT_GENERATOR = """You are an expert networking professional conducting technical interviews. 
Generate a realistic technical interview conversation about networking concepts.
The conversation should match the specified skill level and be between an interviewer and candidate.
Focus on networking concepts, troubleshooting, and real-world scenarios."""

SYSTEM_PROMPT_EVALUATOR = """You are an expert at evaluating technical interviews.
Analyze the given conversation and provide a detailed evaluation of the candidate's performance.
Consider technical accuracy, problem-solving ability, communication skills, and overall competence.
Provide scores and specific feedback based on the demonstrated skill level."""

def get_openai_client() -> OpenAI:
    """Initialize and return OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create a custom httpx client without proxies
    http_client = httpx.Client(
        base_url="https://api.openai.com/v1",
        timeout=60.0,
        follow_redirects=True
    )
    
    return OpenAI(
        api_key=api_key,
        http_client=http_client
    )

def generate_conversation(client: OpenAI, skill_level: str, persona: str) -> List[Dict[str, str]]:
    """Generate a conversation using OpenAI."""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GENERATOR},
            {"role": "user", "content": f"Generate a networking interview conversation for a {skill_level} level candidate. The interviewer persona is {persona}. Include 3-4 technical questions and responses."}
        ]
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse the response into a structured conversation
        conversation_text = completion.choices[0].message.content
        conversation = []
        
        # Split into turns and format
        turns = conversation_text.split("\n\n")
        for turn in turns:
            if ":" in turn:
                role, content = turn.split(":", 1)
                conversation.append({
                    "role": role.strip(),
                    "content": content.strip()
                })
        
        return conversation
    
    except Exception as e:
        print(f"{Colors.RED}Error generating conversation: {str(e)}{Colors.ENDC}")
        return []

def evaluate_conversation(client: OpenAI, conversation: List[Dict[str, str]], skill_level: str) -> Dict[str, Any]:
    """Evaluate a conversation using OpenAI."""
    try:
        # Format conversation for evaluation
        conv_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR},
            {"role": "user", "content": f"Evaluate this {skill_level} level networking interview conversation:\n\n{conv_text}"}
        ]
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse evaluation response
        eval_text = completion.choices[0].message.content
        
        # Extract scores and feedback
        evaluation = {
            "skill_level": skill_level,
            "overall_score": 0,
            "technical_score": 0,
            "communication_score": 0,
            "feedback": eval_text
        }
        
        # Try to parse structured scores if present
        try:
            if "Overall Score:" in eval_text:
                evaluation["overall_score"] = int(eval_text.split("Overall Score:")[1].split("\n")[0].strip().split("/")[0])
            if "Technical Score:" in eval_text:
                evaluation["technical_score"] = int(eval_text.split("Technical Score:")[1].split("\n")[0].strip().split("/")[0])
            if "Communication Score:" in eval_text:
                evaluation["communication_score"] = int(eval_text.split("Communication Score:")[1].split("\n")[0].strip().split("/")[0])
        except:
            pass
        
        return evaluation
    
    except Exception as e:
        print(f"{Colors.RED}Error evaluating conversation: {str(e)}{Colors.ENDC}")
        return {
            "skill_level": skill_level,
            "overall_score": 0,
            "technical_score": 0,
            "communication_score": 0,
            "feedback": "Error during evaluation"
        }

def main():
    """Main function to generate conversations and save to text."""
    print(f"{Colors.HEADER}Starting OpenAI conversation generation...{Colors.ENDC}")
    
    # Create output directory
    output_dir = "simple_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    output_file = os.path.join(output_dir, "conversations.txt")
    debug_file = os.path.join(output_dir, "debug.txt")
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # Open files
    with open(output_file, 'w', encoding='utf-8') as output_f, \
         open(debug_file, 'w', encoding='utf-8') as debug_f:
        
        # Write headers
        output_f.write("NETWORKING CONVERSATIONS (OpenAI Generated)\n")
        output_f.write("=======================================\n\n")
        output_f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        debug_f.write("DEBUG LOG - NETWORKING CONVERSATIONS\n")
        debug_f.write("===================================\n\n")
        debug_f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Set up skill levels
        skill_bases = ['novice', 'intermediate', 'advanced']
        gradients = ['low', 'basic', 'high']
        persona = "INTERVIEWER"
        
        # Track stats
        total = 0
        successful = 0
        
        # Generate and evaluate conversations
        for base in skill_bases:
            for gradient in gradients:
                skill_level = f"{base}_{gradient}"
                
                for i in range(1, 4):
                    total += 1
                    print(f"{Colors.BLUE}Generating: {skill_level.upper()} - #{i}{Colors.ENDC}")
                    
                    # Write conversation header
                    output_f.write(f"\n{'=' * 80}\n")
                    output_f.write(f"{skill_level.upper()} - CONVERSATION {i}\n")
                    output_f.write(f"{'=' * 80}\n\n")
                    
                    # Generate conversation
                    conversation = generate_conversation(client, skill_level, persona)
                    if conversation:
                        # Write conversation
                        output_f.write("CONVERSATION:\n\n")
                        for msg in conversation:
                            output_f.write(f"{msg['role']}: {msg['content']}\n\n")
                        
                        # Evaluate conversation
                        evaluation = evaluate_conversation(client, conversation, skill_level)
                        
                        # Write evaluation
                        output_f.write("\nEVALUATION:\n")
                        output_f.write(f"Overall Score: {evaluation['overall_score']}/10\n")
                        output_f.write(f"Technical Score: {evaluation['technical_score']}/10\n")
                        output_f.write(f"Communication Score: {evaluation['communication_score']}/10\n")
                        output_f.write(f"\nFeedback:\n{evaluation['feedback']}\n\n")
                        
                        successful += 1
                        print(f"{Colors.GREEN}Successfully generated and evaluated.{Colors.ENDC}")
                    else:
                        output_f.write("Failed to generate conversation.\n\n")
                        print(f"{Colors.RED}Failed to generate conversation.{Colors.ENDC}")
                    
                    # Add delay to avoid rate limits
                    time.sleep(2)
        
        # Write summary
        summary = f"\nGeneration Summary:\n" + \
                 f"Total conversations: {total}\n" + \
                 f"Successful conversations: {successful}\n" + \
                 f"Failed conversations: {total - successful}\n"
        
        output_f.write(summary)
        debug_f.write(summary)
        
        print(summary)
        print(f"{Colors.GREEN}Process complete!{Colors.ENDC}")
        print(f"Output saved to: {output_file}")
        print(f"Debug log saved to: {debug_file}")

if __name__ == "__main__":
    main() 