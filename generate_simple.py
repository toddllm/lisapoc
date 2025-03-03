#!/usr/bin/env python3
"""
Simple script to generate conversations and evaluations using direct OpenAI API calls.
Enhanced with comprehensive evaluation framework based on evaluation_design.md.
"""

import os
import sys
import json
import time
import re
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
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

SYSTEM_PROMPT_EVALUATOR = """You are an expert at evaluating networking conversations.

Analyze the given conversation and provide a detailed evaluation based on the following framework:

1. Evaluate each conversation stage (0-3 points each):
   - Opener: Initial conversation starter
   - Carrying Conversation: Maintaining dialogue flow
   - LinkedIn Connection: Asking for professional connection
   - Move On: Gracefully transitioning away
   - Farewell: Closing the conversation

   Scoring guide:
   - 3 points: Optimal response (examples: "What brings you here today?", "What got you started in that?")
   - 2 points: Good response (examples: "How's it going?", "That sounds interesting")
   - 0-1 points: Needs improvement

2. Evaluate across three dimensions:
   - Critical Thinking: How effectively conversational strategies are deployed
   - Communication: Quality of expression and dialogue flow
   - Emotional Intelligence: Ability to read and respond to social cues

3. Calculate dimension scores using these weights:
   - Critical Thinking = (opener × 0.4) + (carrying × 0.3) + (linkedin × 0.6) + (move_on × 0.2) + (farewell × 0.1)
   - Communication = (opener × 0.3) + (carrying × 0.5) + (linkedin × 0.3) + (move_on × 0.2) + (farewell × 0.5)
   - Emotional Intelligence = (opener × 0.3) + (carrying × 0.2) + (linkedin × 0.1) + (move_on × 0.6) + (farewell × 0.4)

4. Consider the skill level in your evaluation:
   - Novice: Expect basic networking skills with some awkwardness and missed opportunities
   - Intermediate: Expect solid networking skills with good flow but some room for improvement
   - Advanced: Expect polished networking skills with excellent conversation management

Format your response with these exact sections:
STAGE SCORES:
Opener: X/3 points
[Brief explanation]
Carrying Conversation: X/3 points
[Brief explanation]
LinkedIn Connection: X/3 points
[Brief explanation]
Move On: X/3 points
[Brief explanation]
Farewell: X/3 points
[Brief explanation]

DIMENSION SCORES:
Critical Thinking: X.X/5 points
[Specific feedback]
Communication: X.X/5 points
[Specific feedback]
Emotional Intelligence: X.X/5 points
[Specific feedback]

OVERALL ASSESSMENT:
Total Score: XX/15 points
Badge Level: [Bronze/Silver/Gold]

STRENGTHS:
- [Specific strength 1]
- [Specific strength 2]
- [Specific strength 3]

AREAS FOR IMPROVEMENT:
- [Specific improvement area 1]
- [Specific improvement area 2]
- [Specific improvement area 3]

ACTIONABLE SUGGESTIONS:
- [Concrete suggestion 1]
- [Concrete suggestion 2]
- [Concrete suggestion 3]
"""

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

def analyze_conversation_stages(conversation: List[Dict[str, str]]) -> Dict[str, List[int]]:
    """
    Analyze a conversation to identify and label different stages.
    
    Args:
        conversation: List of conversation messages with role and content
        
    Returns:
        Dictionary mapping stages to lists of message indices
    """
    stages = {
        'opener': [],
        'carrying_conversation': [],
        'linkedin_connection': [],
        'move_on': [],
        'farewell': []
    }
    
    # Simple heuristic: first exchange is opener, last is farewell
    if len(conversation) >= 2:
        stages['opener'] = [0, 1]  # First two messages
    
    if len(conversation) >= 4:
        # Look for LinkedIn mentions
        for i, msg in enumerate(conversation):
            content = msg['content'].lower()
            if 'linkedin' in content or 'connect' in content:
                stages['linkedin_connection'].append(i)
                # Messages before LinkedIn but after opener are carrying conversation
                carrying_indices = list(range(2, i))
                if carrying_indices:
                    stages['carrying_conversation'] = carrying_indices
                break
    
    # Look for move on signals
    for i, msg in enumerate(conversation):
        content = msg['content'].lower()
        if ('excuse' in content or 'leave' in content or 'go' in content) and i > 2:
            stages['move_on'].append(i)
            # If we found move on, the last 2 messages are farewell
            if i < len(conversation) - 2:
                stages['farewell'] = list(range(i+1, len(conversation)))
            break
    
    # If we didn't find specific stages, make reasonable assumptions
    if not stages['carrying_conversation'] and len(conversation) > 4:
        # Middle messages are carrying conversation
        middle_start = 2
        middle_end = len(conversation) - 2
        stages['carrying_conversation'] = list(range(middle_start, middle_end))
    
    if not stages['farewell'] and len(conversation) >= 2:
        # Last two messages are farewell
        stages['farewell'] = [len(conversation)-2, len(conversation)-1]
    
    return stages

def calculate_dimension_scores(stage_scores: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate dimension scores based on stage scores and predefined weights.
    
    Args:
        stage_scores: Dictionary of scores for each conversation stage
        
    Returns:
        Dictionary of scores for each dimension
    """
    # Weights from evaluation_design.md
    weights = {
        'critical_thinking': {
            'opener': 0.4,
            'carrying_conversation': 0.3,
            'linkedin_connection': 0.6,
            'move_on': 0.2,
            'farewell': 0.1
        },
        'communication': {
            'opener': 0.3,
            'carrying_conversation': 0.5,
            'linkedin_connection': 0.3,
            'move_on': 0.2,
            'farewell': 0.5
        },
        'emotional_intelligence': {
            'opener': 0.3,
            'carrying_conversation': 0.2,
            'linkedin_connection': 0.1,
            'move_on': 0.6,
            'farewell': 0.4
        }
    }
    
    dimension_scores = {}
    
    for dimension, stage_weights in weights.items():
        score = 0.0
        for stage, weight in stage_weights.items():
            if stage in stage_scores:
                score += stage_scores[stage] * weight
        dimension_scores[dimension] = round(score, 1)
    
    return dimension_scores

def determine_badge_level(dimension_scores: Dict[str, float], total_score: int, skill_level: str) -> str:
    """
    Determine the appropriate badge level based on dimension scores, total score, and skill level.
    
    Args:
        dimension_scores: Dictionary of scores for each dimension
        total_score: Total score across all stages
        skill_level: The skill level of the conversation (novice_*, intermediate_*, advanced_*)
        
    Returns:
        Badge level (Bronze, Silver, Gold, or No Badge)
    """
    # Extract dimension scores
    critical_thinking = dimension_scores.get('critical_thinking', 0)
    communication = dimension_scores.get('communication', 0)
    emotional_intelligence = dimension_scores.get('emotional_intelligence', 0)
    
    # Extract skill base (novice, intermediate, advanced)
    skill_base = skill_level.split('_')[0].lower() if '_' in skill_level else skill_level.lower()
    
    # Adjust badge thresholds based on skill level
    if skill_base == 'novice':
        # Novices should mostly get Bronze, occasionally Silver
        if (total_score >= 12 and 
            critical_thinking >= 4 and 
            communication >= 4 and 
            emotional_intelligence >= 4):
            return "Silver"  # Exceptional novice
        elif (total_score >= 5 and 
              critical_thinking >= 1 and 
              communication >= 1 and 
              emotional_intelligence >= 1):
            return "Bronze"  # Standard novice
        else:
            return "No Badge"
            
    elif skill_base == 'intermediate':
        # Intermediates should mostly get Silver, occasionally Gold or Bronze
        if (total_score >= 12 and 
            critical_thinking >= 4 and 
            communication >= 4 and 
            emotional_intelligence >= 4):
            return "Gold"  # Exceptional intermediate
        elif (total_score >= 7 and 
              critical_thinking >= 2 and 
              communication >= 2 and 
              emotional_intelligence >= 2):
            return "Silver"  # Standard intermediate
        elif (total_score >= 3):
            return "Bronze"  # Struggling intermediate
        else:
            return "No Badge"
            
    elif skill_base == 'advanced':
        # Advanced should mostly get Gold, occasionally Silver
        if (total_score >= 10 and 
            critical_thinking >= 3 and 
            communication >= 3 and 
            emotional_intelligence >= 3):
            return "Gold"  # Standard advanced
        elif (total_score >= 5):
            return "Silver"  # Struggling advanced
        else:
            return "Bronze"  # Very poor advanced
    
    # Default fallback using original logic
    if (total_score >= 11 and 
        critical_thinking >= 5 and 
        communication >= 5 and 
        emotional_intelligence >= 5):
        return "Gold"
    elif (total_score >= 6 and 
          critical_thinking >= 3 and 
          communication >= 3 and 
          emotional_intelligence >= 3):
        return "Silver"
    elif (total_score >= 1 and 
          critical_thinking >= 1 and 
          communication >= 1 and 
          emotional_intelligence >= 1):
        return "Bronze"
    else:
        return "No Badge"

def parse_evaluation_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the evaluation response from OpenAI into a structured format.
    
    Args:
        response_text: The text response from the OpenAI API
        
    Returns:
        Dictionary containing parsed evaluation data
    """
    evaluation = {
        'stage_scores': {},
        'dimension_scores': {},
        'strengths': [],
        'areas_for_improvement': [],
        'actionable_suggestions': [],
        'total_score': 0,
        'badge_level': 'No Badge'
    }
    
    # Extract stage scores
    stage_score_pattern = r'(Opener|Carrying Conversation|LinkedIn Connection|Move On|Farewell): (\d+)/3 points'
    stage_matches = re.finditer(stage_score_pattern, response_text)
    for match in stage_matches:
        stage = match.group(1).lower().replace(' ', '_')
        score = int(match.group(2))
        evaluation['stage_scores'][stage] = score
    
    # Extract dimension scores
    dimension_score_pattern = r'(Critical Thinking|Communication|Emotional Intelligence): (\d+\.\d+|\d+)/5 points'
    dimension_matches = re.finditer(dimension_score_pattern, response_text)
    for match in dimension_matches:
        dimension = match.group(1).lower().replace(' ', '_')
        score = float(match.group(2))
        evaluation['dimension_scores'][dimension] = score
    
    # Extract total score
    total_score_match = re.search(r'Total Score: (\d+)/15 points', response_text)
    if total_score_match:
        evaluation['total_score'] = int(total_score_match.group(1))
    
    # Extract badge level
    badge_match = re.search(r'Badge Level: (Bronze|Silver|Gold|No Badge)', response_text)
    if badge_match:
        evaluation['badge_level'] = badge_match.group(1)
    
    # Extract strengths
    strengths_section = re.search(r'STRENGTHS:\s*(.+?)(?=AREAS FOR IMPROVEMENT:|$)', response_text, re.DOTALL)
    if strengths_section:
        strengths_text = strengths_section.group(1).strip()
        strengths = re.findall(r'- (.+)', strengths_text)
        evaluation['strengths'] = [s.strip() for s in strengths]
    
    # Extract areas for improvement
    improvements_section = re.search(r'AREAS FOR IMPROVEMENT:\s*(.+?)(?=ACTIONABLE SUGGESTIONS:|$)', response_text, re.DOTALL)
    if improvements_section:
        improvements_text = improvements_section.group(1).strip()
        improvements = re.findall(r'- (.+)', improvements_text)
        evaluation['areas_for_improvement'] = [i.strip() for i in improvements]
    
    # Extract actionable suggestions
    suggestions_section = re.search(r'ACTIONABLE SUGGESTIONS:\s*(.+?)(?=$)', response_text, re.DOTALL)
    if suggestions_section:
        suggestions_text = suggestions_section.group(1).strip()
        suggestions = re.findall(r'- (.+)', suggestions_text)
        evaluation['actionable_suggestions'] = [s.strip() for s in suggestions]
    
    return evaluation

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
            max_tokens=2000
        )
        
        # Parse evaluation response
        eval_text = completion.choices[0].message.content
        evaluation = parse_evaluation_response(eval_text)
        
        # If stage scores weren't properly extracted, try to analyze the conversation
        if not evaluation['stage_scores']:
            print(f"{Colors.YELLOW}Warning: Stage scores not found in evaluation. Analyzing conversation stages...{Colors.ENDC}")
            stages = analyze_conversation_stages(conversation)
            # This is just a placeholder - in a real implementation, we would score each stage
            evaluation['stage_scores'] = {
                'opener': 2,
                'carrying_conversation': 2,
                'linkedin_connection': 2,
                'move_on': 2,
                'farewell': 2
            }
        
        # If dimension scores weren't properly extracted, calculate them
        if not evaluation['dimension_scores']:
            print(f"{Colors.YELLOW}Warning: Dimension scores not found in evaluation. Calculating from stage scores...{Colors.ENDC}")
            evaluation['dimension_scores'] = calculate_dimension_scores(evaluation['stage_scores'])
        
        # Calculate total score if not provided
        if not evaluation['total_score']:
            evaluation['total_score'] = sum(evaluation['stage_scores'].values())
        
        # Determine badge level if not provided or override based on skill level
        evaluation['badge_level'] = determine_badge_level(
            evaluation['dimension_scores'], 
            evaluation['total_score'],
            skill_level
        )
        
        # Add the raw evaluation text for reference
        evaluation['raw_evaluation'] = eval_text
        
        return evaluation
    
    except Exception as e:
        print(f"{Colors.RED}Error evaluating conversation: {str(e)}{Colors.ENDC}")
        traceback.print_exc()
        return {
            'stage_scores': {
                'opener': 1,
                'carrying_conversation': 1,
                'linkedin_connection': 1,
                'move_on': 1,
                'farewell': 1
            },
            'dimension_scores': {
                'critical_thinking': 1.0,
                'communication': 1.0,
                'emotional_intelligence': 1.0
            },
            'strengths': ["Unable to evaluate strengths due to an error"],
            'areas_for_improvement': ["Unable to evaluate areas for improvement due to an error"],
            'actionable_suggestions': ["Try again with a different conversation"],
            'total_score': 5,
            'badge_level': "Bronze",
            'error': str(e)
        }

def format_evaluation_for_output(evaluation: Dict[str, Any]) -> str:
    """
    Format the evaluation dictionary into a readable string for output.
    
    Args:
        evaluation: Dictionary containing evaluation data
        
    Returns:
        Formatted string representation of the evaluation
    """
    output = "EVALUATION:\n"
    output += "===========\n\n"
    
    # Stage Scores
    output += "STAGE SCORES:\n"
    output += "------------\n"
    for stage, score in evaluation['stage_scores'].items():
        stage_name = stage.replace('_', ' ').title()
        output += f"{stage_name}: {score}/3 points\n"
    output += "\n"
    
    # Dimension Scores
    output += "DIMENSION SCORES:\n"
    output += "---------------\n"
    for dimension, score in evaluation['dimension_scores'].items():
        dimension_name = dimension.replace('_', ' ').title()
        output += f"{dimension_name}: {score}/5 points\n"
    output += "\n"
    
    # Overall Assessment
    output += "OVERALL ASSESSMENT:\n"
    output += "-----------------\n"
    output += f"Total Score: {evaluation['total_score']}/15 points\n"
    output += f"Badge Level: {evaluation['badge_level']}\n\n"
    
    # Strengths
    output += "STRENGTHS:\n"
    output += "---------\n"
    for strength in evaluation['strengths']:
        output += f"- {strength}\n"
    output += "\n"
    
    # Areas for Improvement
    output += "AREAS FOR IMPROVEMENT:\n"
    output += "--------------------\n"
    for area in evaluation['areas_for_improvement']:
        output += f"- {area}\n"
    output += "\n"
    
    # Actionable Suggestions
    output += "ACTIONABLE SUGGESTIONS:\n"
    output += "---------------------\n"
    for suggestion in evaluation['actionable_suggestions']:
        output += f"- {suggestion}\n"
    output += "\n"
    
    return output

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
                        
                        # Debug log
                        debug_f.write(f"\n{'-' * 40}\n")
                        debug_f.write(f"DEBUG: {skill_level.upper()} - #{i}\n")
                        debug_f.write(f"{'-' * 40}\n")
                        debug_f.write(f"Conversation length: {len(conversation)} messages\n")
                        
                        # Analyze conversation stages
                        stages = analyze_conversation_stages(conversation)
                        debug_f.write("Detected stages:\n")
                        for stage, indices in stages.items():
                            debug_f.write(f"  {stage}: {indices}\n")
                        
                        # Evaluate conversation
                        print(f"{Colors.CYAN}Evaluating conversation...{Colors.ENDC}")
                        evaluation = evaluate_conversation(client, conversation, skill_level)
                        
                        # Debug log evaluation
                        debug_f.write("\nEvaluation results:\n")
                        debug_f.write(f"  Stage scores: {evaluation['stage_scores']}\n")
                        debug_f.write(f"  Dimension scores: {evaluation['dimension_scores']}\n")
                        debug_f.write(f"  Total score: {evaluation['total_score']}\n")
                        debug_f.write(f"  Badge level: {evaluation['badge_level']}\n")
                        debug_f.write(f"  Skill level: {skill_level}\n")
                        
                        # Format and write evaluation
                        formatted_evaluation = format_evaluation_for_output(evaluation)
                        output_f.write("\n" + formatted_evaluation + "\n")
                        
                        successful += 1
                        print(f"{Colors.GREEN}Successfully generated and evaluated.{Colors.ENDC}")
                        print(f"{Colors.GREEN}Badge Level: {evaluation['badge_level']}{Colors.ENDC}")
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