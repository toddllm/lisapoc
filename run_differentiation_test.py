#!/usr/bin/env python3
"""
Run a test to verify the improved differentiation between skill levels.
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_conversation_gpt import ConversationGenerator, ConversationEvaluator

# ANSI colors for better terminal output
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

def run_differentiation_test():
    """Run a test to verify the improved differentiation between skill levels."""
    print(f"{Colors.HEADER}{Colors.BOLD}\n" + "="*80)
    print("TESTING IMPROVED DIFFERENTIATION BETWEEN SKILL LEVELS")
    print("="*80 + f"{Colors.ENDC}")
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{Colors.RED}Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here{Colors.ENDC}")
        return False
    
    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")
    
    # Define skill levels to test
    skill_levels = ["novice", "intermediate", "advanced"]
    
    print(f"\n{Colors.CYAN}Testing differentiation between {len(skill_levels)} skill levels{Colors.ENDC}")
    
    # Create output directory
    output_dir = "differentiation_test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize generator and evaluator
    print(f"\n{Colors.CYAN}Initializing conversation generator and evaluator...{Colors.ENDC}")
    
    generator = ConversationGenerator()
    evaluator = ConversationEvaluator()
    
    print("Generator and evaluator initialized.")
    
    results = []
    
    # Generate and evaluate a conversation for each skill level
    for i, skill_level in enumerate(skill_levels):
        print(f"\n{Colors.BOLD}" + "-"*80)
        print(f"{Colors.BLUE}TESTING SKILL LEVEL {i+1}/{len(skill_levels)}: {skill_level.upper()}{Colors.ENDC}")
        print("-"*80)
        
        # Generate conversation
        print(f"\n{Colors.CYAN}[1/2] Generating conversation for {skill_level} skill level...{Colors.ENDC}")
        print("      Making API call to OpenAI (this may take 15-30 seconds)...")
        
        start_time = time.time()
        conversation = generator.generate_conversation("jake", skill_level)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n{Colors.GREEN}      Conversation generated in {duration:.2f} seconds{Colors.ENDC}")
        print(f"      Conversation has {len(conversation)} exchanges")
        
        # Display sample of conversation
        print(f"\n{Colors.CYAN}      SAMPLE EXCHANGE:{Colors.ENDC}")
        if conversation:
            exchange = conversation[0]
            print(f"      Stage: {exchange.get('stage', 'unknown').upper()}")
            print(f"      AI: {exchange.get('ai_prompt', '')}")
            print(f"      User: {exchange.get('user_response', '')}")
        
        # Evaluate conversation
        print(f"\n{Colors.CYAN}[2/2] Evaluating conversation...{Colors.ENDC}")
        print(f"      This requires {len(conversation)} API calls to OpenAI (one per exchange)...")
        
        start_time = time.time()
        evaluation = evaluator.evaluate_conversation(conversation)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n{Colors.GREEN}      Evaluation completed in {duration:.2f} seconds{Colors.ENDC}")
        
        # Show badge with color
        badge_level = evaluation.get('badge_level', 'Bronze')
        badge_color = {
            'Bronze': Colors.RED,
            'Silver': Colors.CYAN,
            'Gold': Colors.YELLOW
        }.get(badge_level, Colors.ENDC)
        
        print("\n      EVALUATION RESULTS:")
        print(f"      - Total Score: {evaluation.get('total_score', 0)}")
        print(f"      - Percentile Score: {evaluation.get('percentile_score', 0)}%")
        print(f"      - Badge Level: {badge_color}{badge_level}{Colors.ENDC}")
        
        print("\n      Skill Scores:")
        for skill, score in evaluation.get('skill_scores', {}).items():
            skill_badge = evaluation.get('badges', {}).get(skill, 'Bronze')
            skill_color = {
                'Bronze': Colors.RED,
                'Silver': Colors.CYAN, 
                'Gold': Colors.YELLOW
            }.get(skill_badge, Colors.ENDC)
            
            print(f"      - {skill.replace('_', ' ').title()}: {score} ({skill_color}{skill_badge}{Colors.ENDC})")
        
        # Store result
        result = {
            "skill_level": skill_level,
            "conversation": conversation,
            "evaluation": evaluation
        }
        
        results.append(result)
        
        # Save result
        filename = f"{output_dir}/{skill_level}_conversation.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{Colors.GREEN}Results saved to {filename}{Colors.ENDC}")
    
    # Create summary table
    summary_data = []
    for result in results:
        eval_data = result["evaluation"]
        summary_data.append({
            "skill_level": result["skill_level"],
            "total_score": eval_data.get("total_score", 0),
            "percentile_score": eval_data.get("percentile_score", 0),
            "badge_level": eval_data.get("badge_level", "Bronze"),
            "critical_thinking": eval_data.get("skill_scores", {}).get("critical_thinking", 0),
            "communication": eval_data.get("skill_scores", {}).get("communication", 0),
            "emotional_intelligence": eval_data.get("skill_scores", {}).get("emotional_intelligence", 0)
        })
    
    df = pd.DataFrame(summary_data)
    
    # Print summary
    print(f"\n{Colors.BOLD}" + "-"*80)
    print(f"{Colors.BLUE}DIFFERENTIATION SUMMARY{Colors.ENDC}")
    print("-"*80)
    print("\nResults by skill level:")
    
    for index, row in df.iterrows():
        badge_color = {
            'Bronze': Colors.RED,
            'Silver': Colors.CYAN,
            'Gold': Colors.YELLOW
        }.get(row['badge_level'], Colors.ENDC)
        
        print(f"\n  {row['skill_level'].upper()}:")
        print(f"    Total Score: {row['total_score']}")
        print(f"    Percentile Score: {row['percentile_score']}%")
        print(f"    Badge Level: {badge_color}{row['badge_level']}{Colors.ENDC}")
        print(f"    Critical Thinking: {row['critical_thinking']}")
        print(f"    Communication: {row['communication']}")
        print(f"    Emotional Intelligence: {row['emotional_intelligence']}")
    
    # Visualize results
    try:
        # Skill scores
        plt.figure(figsize=(10, 6))
        plt.bar(
            [f"{level.capitalize()}\nCritical Thinking" for level in skill_levels],
            df['critical_thinking'],
            color='#3498db'
        )
        plt.bar(
            [f"{level.capitalize()}\nCommunication" for level in skill_levels],
            df['communication'],
            color='#2ecc71'
        )
        plt.bar(
            [f"{level.capitalize()}\nEmotional Intelligence" for level in skill_levels],
            df['emotional_intelligence'],
            color='#e74c3c'
        )
        plt.title('Skill Scores by Level')
        plt.ylabel('Score')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{output_dir}/skill_scores_comparison.png")
        print(f"\n{Colors.GREEN}Visualization saved to {output_dir}/skill_scores_comparison.png{Colors.ENDC}")
        
        # Total scores
        plt.figure(figsize=(8, 6))
        plt.bar(
            [level.capitalize() for level in skill_levels],
            df['total_score'],
            color=['#e74c3c', '#3498db', '#2ecc71']
        )
        plt.title('Total Scores by Skill Level')
        plt.ylabel('Score')
        
        # Add data labels
        for i, v in enumerate(df['total_score']):
            plt.text(i, v + 0.5, str(v), ha='center')
        
        # Save the figure
        plt.savefig(f"{output_dir}/total_scores_comparison.png")
        print(f"{Colors.GREEN}Visualization saved to {output_dir}/total_scores_comparison.png{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Error generating visualizations: {e}{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*80)
    print("DIFFERENTIATION TEST COMPLETED")
    print("="*80 + f"{Colors.ENDC}")
    
    # Check for clear differentiation
    if len(df) < 3:
        return True  # Not enough data to check
    
    novice_score = df[df['skill_level'] == 'novice']['total_score'].values[0]
    intermediate_score = df[df['skill_level'] == 'intermediate']['total_score'].values[0]
    advanced_score = df[df['skill_level'] == 'advanced']['total_score'].values[0]
    
    # Check differences
    novice_intermediate_diff = intermediate_score - novice_score
    intermediate_advanced_diff = advanced_score - intermediate_score
    
    print("\nDifferentiation Check:")
    print(f"  Novice to Intermediate Difference: {novice_intermediate_diff}")
    print(f"  Intermediate to Advanced Difference: {intermediate_advanced_diff}")
    
    if novice_intermediate_diff >= 5 and intermediate_advanced_diff >= 3:
        print(f"\n{Colors.GREEN}Good differentiation between skill levels!{Colors.ENDC}")
        return True
    else:
        print(f"\n{Colors.YELLOW}Warning: Differentiation between skill levels could be improved.{Colors.ENDC}")
        return True

if __name__ == "__main__":
    print(f"{Colors.CYAN}Starting differentiation test...{Colors.ENDC}")
    success = run_differentiation_test()
    if success:
        print(f"\n{Colors.GREEN}Test completed successfully!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}Test failed. See output above for details.{Colors.ENDC}")
        sys.exit(1) 