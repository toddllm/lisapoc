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

def determine_badge_level(dimension_scores, total_score, skill_level=None):
    """
    Determine the badge level based on dimension scores and total score.
    
    Args:
        dimension_scores (dict): Dictionary of dimension scores
        total_score (float): Total score
        skill_level (str): Skill level of the user (novice_low, novice_basic, novice_high, etc.)
        
    Returns:
        str: Badge level (Bronze, Silver, or Gold)
    """
    # Extract skill category and gradient from skill_level
    skill_category = None
    gradient = None
    
    if skill_level:
        parts = skill_level.lower().split('_')
        if len(parts) >= 1:
            skill_category = parts[0]  # novice, intermediate, or advanced
        if len(parts) >= 2:
            gradient = parts[1]  # low, basic, or high
    
    # Default thresholds
    bronze_threshold = 6
    silver_threshold = 9
    gold_threshold = 12
    
    # Dimension minimum thresholds
    bronze_dim_min = 1.5
    silver_dim_min = 2.5
    gold_dim_min = 3.5
    
    # Adjust thresholds based on skill category
    if skill_category == 'novice':
        # Novices should mostly get Bronze
        bronze_threshold = 5  # Easier to get Bronze
        silver_threshold = 10  # Harder to get Silver
        gold_threshold = 15   # Very hard to get Gold
        
        # Adjust based on gradient within novice
        if gradient == 'high':
            silver_threshold = 9  # Slightly easier for high novices to get Silver
        elif gradient == 'low':
            bronze_threshold = 4  # Even easier for low novices to get Bronze
            
    elif skill_category == 'intermediate':
        # Intermediates should mostly get Silver
        bronze_threshold = 6
        silver_threshold = 8  # Easier to get Silver
        gold_threshold = 13   # Harder to get Gold
        
        # Adjust based on gradient within intermediate
        if gradient == 'high':
            gold_threshold = 12  # Slightly easier for high intermediates to get Gold
        elif gradient == 'low':
            bronze_threshold = 5  # Easier for low intermediates to avoid Bronze
            
    elif skill_category == 'advanced':
        # Advanced should get Silver or Gold
        bronze_threshold = 7  # Harder to get Bronze
        silver_threshold = 8  # Easy to get Silver
        gold_threshold = 12   # Possible to get Gold
        
        # Adjust based on gradient within advanced
        if gradient == 'high':
            bronze_threshold = 8  # Very hard for high advanced to get Bronze
            gold_threshold = 13   # Harder for high advanced to get Gold (more challenging)
        elif gradient == 'low':
            gold_threshold = 11   # Slightly easier for low advanced to get Gold
    
    # Check dimension minimums
    critical_thinking = dimension_scores.get('critical_thinking', 0)
    communication = dimension_scores.get('communication', 0)
    emotional_intelligence = dimension_scores.get('emotional_intelligence', 0)
    
    # Determine badge level based on total score and dimension minimums
    if total_score >= gold_threshold and all([
        critical_thinking >= gold_dim_min,
        communication >= gold_dim_min,
        emotional_intelligence >= gold_dim_min
    ]):
        return 'Gold'
    elif total_score >= silver_threshold and all([
        critical_thinking >= silver_dim_min,
        communication >= silver_dim_min,
        emotional_intelligence >= silver_dim_min
    ]):
        return 'Silver'
    elif total_score >= bronze_threshold and all([
        critical_thinking >= bronze_dim_min,
        communication >= bronze_dim_min,
        emotional_intelligence >= bronze_dim_min
    ]):
        return 'Bronze'
    else:
        return 'Bronze'  # Default to Bronze for any conversation that doesn't meet minimum criteria

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
    """Main function to generate conversations and save them to a file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('simple_output', exist_ok=True)
        
        # Open files for writing
        with open('simple_output/conversations.txt', 'w') as f, open('simple_output/debug.txt', 'w') as debug_f:
            debug_f.write("Starting conversation generation\n")
            
            # Define skill levels and gradients
            skill_levels = ['novice', 'intermediate', 'advanced']
            gradients = ['low', 'basic', 'high']
            
            # Generate conversations for each skill level and gradient
            for skill_level in skill_levels:
                for gradient in gradients:
                    full_skill_level = f"{skill_level}_{gradient}".upper()
                    debug_f.write(f"\nGenerating conversation for {full_skill_level}\n")
                    
                    try:
                        # Generate conversation
                        conversation = generate_conversation(full_skill_level)
                        debug_f.write(f"Generated conversation for {full_skill_level}\n")
                        
                        # Evaluate conversation
                        evaluation = evaluate_conversation(conversation)
                        debug_f.write(f"Evaluated conversation for {full_skill_level}\n")
                        debug_f.write(f"  Stage scores: {evaluation['stage_scores']}\n")
                        debug_f.write(f"  Dimension scores: {evaluation['dimension_scores']}\n")
                        debug_f.write(f"  Total score: {evaluation['total_score']}\n")
                        
                        # Determine badge level with skill level
                        skill_level_for_badge = f"{skill_level}_{gradient}".lower()
                        badge_level = determine_badge_level(
                            evaluation['dimension_scores'], 
                            evaluation['total_score'],
                            skill_level_for_badge
                        )
                        evaluation['badge_level'] = badge_level
                        
                        debug_f.write(f"  Badge level: {badge_level}\n")
                        debug_f.write(f"  Skill level: {skill_level_for_badge}\n")
                        
                        # Write to file
                        f.write(f"# Conversation for {full_skill_level}\n\n")
                        f.write(conversation)
                        f.write("\n\n")
                        f.write(f"## Evaluation for {full_skill_level}\n\n")
                        f.write(f"Badge Level: {badge_level}\n\n")
                        f.write(f"Total Score: {evaluation['total_score']}\n\n")
                        f.write("Dimension Scores:\n")
                        for dimension, score in evaluation['dimension_scores'].items():
                            f.write(f"- {dimension.replace('_', ' ').title()}: {score}\n")
                        f.write("\n")
                        f.write("Stage Scores:\n")
                        for stage, score in evaluation['stage_scores'].items():
                            f.write(f"- {stage.replace('_', ' ').title()}: {score}\n")
                        f.write("\n")
                        f.write("Feedback:\n")
                        f.write(evaluation['feedback'])
                        f.write("\n\n")
                        f.write("-" * 80)
                        f.write("\n\n")
                        
                    except Exception as e:
                        debug_f.write(f"Error generating or evaluating conversation for {full_skill_level}: {str(e)}\n")
                        traceback.print_exc(file=debug_f)
                        
                        # Use fallback data
                        f.write(f"# Conversation for {full_skill_level} (FALLBACK)\n\n")
                        f.write(FALLBACK_CONVERSATION)
                        f.write("\n\n")
                        f.write(f"## Evaluation for {full_skill_level} (FALLBACK)\n\n")
                        f.write(FALLBACK_EVALUATION)
                        f.write("\n\n")
                        f.write("-" * 80)
                        f.write("\n\n")
            
            debug_f.write("\nFinished generating conversations\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 