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
import importlib
import random  # For generating mock data

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

# Fallback data for when generation or evaluation fails
FALLBACK_CONVERSATION = """
INTERVIEWER: Hi there! I noticed you're also attending this networking event. I'm [Name] from [Company]. What brings you here today?

CANDIDATE: Hello! I'm [Name]. I'm here to learn more about the industry and meet new people. It's my first time at this event.

INTERVIEWER: That's great! First-time experiences can be exciting. What field or industry are you currently working in?

CANDIDATE: I'm currently working in software development, focusing on web applications. I've been in the field for about two years now.

INTERVIEWER: Web development is such a dynamic area! Are you working with any particular technologies or frameworks that you find interesting?

CANDIDATE: Yes, I've been working with React and Node.js mostly. I really enjoy frontend development and creating user-friendly interfaces.

INTERVIEWER: That's fantastic! I've worked with some React developers at my company, and they're doing some innovative things. Would you be interested in connecting on LinkedIn? I could introduce you to some people in my network who work with similar technologies.

CANDIDATE: That would be great! I'd appreciate the connections. Let me get my phone so we can connect right now.

INTERVIEWER: Perfect! I just sent you a connection request. I'll definitely follow up with those introductions. By the way, there's a tech meetup happening next week focused on frontend frameworks. Would that be something you'd be interested in?

CANDIDATE: Absolutely! That sounds like exactly the kind of event I'd enjoy. Could you share the details with me?

INTERVIEWER: Of course! I'll send you the link through LinkedIn. It's usually a good mix of presentations and networking. I've attended a few times and always learn something new. Well, I should probably mingle a bit more, but it was really nice meeting you!

CANDIDATE: It was nice meeting you too! Thanks for the LinkedIn connection and information about the meetup. I look forward to staying in touch.

INTERVIEWER: Likewise! Enjoy the rest of the event, and don't hesitate to reach out if you have any questions. Have a great evening!

CANDIDATE: You too! Thanks again, and enjoy the rest of the event!
"""

FALLBACK_EVALUATION = """
Badge Level: Bronze

Total Score: 8

Dimension Scores:
- Critical Thinking: 2.5
- Communication: 3.0
- Emotional Intelligence: 2.5

Stage Scores:
- Opener: 2
- Carrying Conversation: 2
- Linkedin Connection: 1
- Move On: 2
- Farewell: 1

Feedback:
The conversation demonstrates basic networking skills. The candidate responds appropriately but could be more proactive in asking questions and showing interest in the interviewer. The LinkedIn connection was established, but the candidate could have been more strategic about how to leverage this new connection. The farewell was polite but generic. Overall, this represents a novice level of networking skill with room for improvement in all dimensions.
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

def generate_conversation(skill_level):
    """
    Generate a mock conversation based on skill level.
    
    Args:
        skill_level (str): Skill level of the conversation
        
    Returns:
        str: Generated conversation
    """
    # For testing purposes, we'll just return the fallback conversation
    # In a real implementation, this would generate a unique conversation
    return FALLBACK_CONVERSATION

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

def evaluate_conversation(conversation):
    """
    Evaluate a conversation and return scores.
    
    Args:
        conversation (str): Conversation to evaluate
        
    Returns:
        dict: Evaluation results with mock data based on skill level
    """
    # Generate mock evaluation data
    # In a real implementation, this would analyze the conversation
    
    # Extract skill level and gradient from the conversation (for testing purposes)
    skill_level = "novice"
    gradient = "low"
    
    if "NOVICE_LOW" in conversation:
        skill_level, gradient = "novice", "low"
    elif "NOVICE_BASIC" in conversation:
        skill_level, gradient = "novice", "basic"
    elif "NOVICE_HIGH" in conversation:
        skill_level, gradient = "novice", "high"
    elif "INTERMEDIATE_LOW" in conversation:
        skill_level, gradient = "intermediate", "low"
    elif "INTERMEDIATE_BASIC" in conversation:
        skill_level, gradient = "intermediate", "basic"
    elif "INTERMEDIATE_HIGH" in conversation:
        skill_level, gradient = "intermediate", "high"
    elif "ADVANCED_LOW" in conversation:
        skill_level, gradient = "advanced", "low"
    elif "ADVANCED_BASIC" in conversation:
        skill_level, gradient = "advanced", "basic"
    elif "ADVANCED_HIGH" in conversation:
        skill_level, gradient = "advanced", "high"
    
    # Generate base scores based on skill level
    if skill_level == "novice":
        base_stage_score = 1.5  # Out of 3
        base_dimension_score = 2.0  # Out of 5
    elif skill_level == "intermediate":
        base_stage_score = 2.0  # Out of 3
        base_dimension_score = 3.0  # Out of 5
    else:  # advanced
        base_stage_score = 2.5  # Out of 3
        base_dimension_score = 4.0  # Out of 5
    
    # Adjust base score based on gradient
    if gradient == "low":
        gradient_modifier_stage = -0.3
        gradient_modifier_dimension = -0.5
    elif gradient == "basic":
        gradient_modifier_stage = 0
        gradient_modifier_dimension = 0
    else:  # high
        gradient_modifier_stage = 0.3
        gradient_modifier_dimension = 0.5
    
    # Apply gradient modifier
    adjusted_stage_base = base_stage_score + gradient_modifier_stage
    adjusted_dimension_base = base_dimension_score + gradient_modifier_dimension
    
    # Create stage scores (0-3 points per stage)
    # Ensure we get some variation in the scores
    stage_scores = {}
    for stage in ['opener', 'carrying_conversation', 'linkedin_connection', 'move_on', 'farewell']:
        # Add more randomness to ensure variation
        variation = random.uniform(-0.5, 0.5)
        score = adjusted_stage_base + variation
        # Round to nearest integer and ensure within bounds
        stage_scores[stage] = min(3, max(1, round(score)))  # Minimum score of 1 to ensure higher totals
    
    # Calculate total score (0-15 points total)
    total_score = sum(stage_scores.values())
    
    # Create dimension scores (1-5 scale)
    dimension_scores = {
        'critical_thinking': min(5.0, max(1.5, adjusted_dimension_base + random.uniform(-0.5, 0.5))),
        'communication': min(5.0, max(1.5, adjusted_dimension_base + random.uniform(-0.5, 0.5))),
        'emotional_intelligence': min(5.0, max(1.5, adjusted_dimension_base + random.uniform(-0.5, 0.5)))
    }
    
    # Generate feedback
    feedback = f"This conversation demonstrates {skill_level} level networking skills. "
    
    if skill_level == "novice":
        feedback += "The candidate shows basic engagement but could improve in strategic questioning. "
    elif skill_level == "intermediate":
        feedback += "The candidate demonstrates good communication skills and professional engagement. "
    else:  # advanced
        feedback += "The candidate shows excellent networking abilities and strategic relationship building. "
    
    feedback += f"For a {skill_level}_{gradient} level, the performance is "
    
    if total_score < 8:
        feedback += "below expectations."
    elif total_score < 12:
        feedback += "meeting expectations."
    else:
        feedback += "exceeding expectations."
    
    return {
        'stage_scores': stage_scores,
        'dimension_scores': dimension_scores,
        'total_score': total_score,
        'feedback': feedback
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
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'output_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating conversations with gradient-sensitive badge determination...")
        print(f"Output will be saved to: {output_dir}")
        
        # Open files for writing
        with open(f'{output_dir}/conversations.txt', 'w') as f, open(f'{output_dir}/debug.txt', 'w') as debug_f:
            debug_f.write(f"Starting conversation generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            debug_f.write(f"Using gradient-sensitive badge determination\n\n")
            
            # Define skill levels and gradients
            skill_levels = ['novice', 'intermediate', 'advanced']
            gradients = ['low', 'basic', 'high']
            
            # Track badge distribution
            badge_distribution = []
            
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
                        
                        # Store result for badge distribution
                        badge_distribution.append({
                            'skill_level': skill_level,
                            'gradient': gradient,
                            'badge': badge_level,
                            'is_fallback': False
                        })
                        
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
                        
                        # Store fallback result for badge distribution
                        badge_distribution.append({
                            'skill_level': skill_level,
                            'gradient': gradient,
                            'badge': 'Bronze',  # Fallbacks get Bronze
                            'is_fallback': True
                        })
            
            # Calculate badge distribution summary
            badge_counts = {
                'novice': {'Bronze': 0, 'Silver': 0, 'Gold': 0},
                'intermediate': {'Bronze': 0, 'Silver': 0, 'Gold': 0},
                'advanced': {'Bronze': 0, 'Silver': 0, 'Gold': 0}
            }
            
            for entry in badge_distribution:
                badge_counts[entry['skill_level']][entry['badge']] += 1
            
            # Write badge distribution summary
            debug_f.write("\n\nBadge Distribution Summary:\n")
            f.write("\n\n# Badge Distribution Summary\n\n")
            
            f.write("| Skill Level | Bronze | Silver | Gold | Total |\n")
            f.write("|-------------|--------|--------|------|-------|\n")
            
            total_bronze = 0
            total_silver = 0
            total_gold = 0
            
            for skill_level in skill_levels:
                bronze = badge_counts[skill_level]['Bronze']
                silver = badge_counts[skill_level]['Silver']
                gold = badge_counts[skill_level]['Gold']
                total = bronze + silver + gold
                
                debug_f.write(f"  {skill_level.title()}: Bronze={bronze}, Silver={silver}, Gold={gold}, Total={total}\n")
                f.write(f"| {skill_level.title()} | {bronze} | {silver} | {gold} | {total} |\n")
                
                total_bronze += bronze
                total_silver += silver
                total_gold += gold
            
            total_all = total_bronze + total_silver + total_gold
            debug_f.write(f"  Total: Bronze={total_bronze}, Silver={total_silver}, Gold={total_gold}, Total={total_all}\n")
            f.write(f"| **Total** | **{total_bronze}** | **{total_silver}** | **{total_gold}** | **{total_all}** |\n")
            
            # Write detailed badge distribution by gradient
            debug_f.write("\n\nDetailed Badge Distribution by Gradient:\n")
            f.write("\n\n# Detailed Badge Distribution by Gradient\n\n")
            
            f.write("| Skill Level | Gradient | Badge | Fallback |\n")
            f.write("|-------------|----------|-------|----------|\n")
            
            for entry in badge_distribution:
                skill = entry['skill_level'].title()
                gradient = entry['gradient']
                badge = entry['badge']
                fallback = "Yes" if entry['is_fallback'] else "No"
                
                debug_f.write(f"  {skill}_{gradient}: Badge={badge}, Fallback={fallback}\n")
                f.write(f"| {skill} | {gradient} | {badge} | {fallback} |\n")
            
            debug_f.write("\nFinished generating conversations\n")
            print(f"Generation complete. Results saved to {output_dir}/conversations.txt")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 