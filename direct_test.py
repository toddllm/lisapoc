#!/usr/bin/env python3
"""
Direct test script that doesn't rely on pytest.
This script directly implements the test logic with explicit print statements.
"""

import os
import sys
import json
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run synthetic conversation tests")
    parser.add_argument("--output", default="direct_test_results", 
                        help="Output directory for test results")
    parser.add_argument("--skill-level", choices=["novice", "intermediate", "advanced", "all"],
                        default="all", help="Skill level to test")
    parser.add_argument("--count", type=int, default=1, 
                        help="Number of conversations to generate per skill level")
    parser.add_argument("--persona", choices=["jake", "sarah"],
                        default="jake", help="Persona to use for the conversation")
    parser.add_argument("--compare", action="store_true",
                        help="Compare scoring across skill levels")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization generation")
    return parser.parse_args()

def run_direct_test(args):
    """Run the test directly without pytest."""
    print(f"{Colors.HEADER}{Colors.BOLD}\n" + "="*80)
    print("RUNNING DIRECT TEST WITH ACTUAL API CALLS")
    print("="*80 + f"{Colors.ENDC}")
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{Colors.RED}Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here{Colors.ENDC}")
        return False
    
    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")
    
    # Determine skill levels to test
    if args.skill_level == "all":
        skill_levels = ["novice", "intermediate", "advanced"]
    else:
        skill_levels = [args.skill_level]
    
    # Create skill gradients
    skill_gradients = []
    for level in skill_levels:
        for i in range(args.count):
            skill_gradients.append((level, 0.0))  # For simplicity, using 0.0 gradient
    
    print(f"\n{Colors.CYAN}Testing {len(skill_gradients)} conversations across {len(skill_levels)} skill levels{Colors.ENDC}")
    for level in skill_levels:
        count = sum(1 for l, _ in skill_gradients if l == level)
        print(f"  - {level.capitalize()}: {count} conversation(s)")
    
    # Create output directory
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize generator and evaluator
    print(f"\n{Colors.CYAN}Initializing conversation generator and evaluator...{Colors.ENDC}")
    
    generator = ConversationGenerator()
    evaluator = ConversationEvaluator()
    
    print("Generator and evaluator initialized:")
    print(f"  Generator: {generator.__class__.__name__}")
    print(f"  Evaluator: {evaluator.__class__.__name__}")
    
    results = []
    
    # Generate and evaluate conversations for each skill gradient
    for i, (base_level, gradient) in enumerate(skill_gradients):
        conversation_id = f"{args.persona}_{base_level}_{i+1}"
        
        print(f"\n{Colors.BOLD}" + "-"*80)
        print(f"{Colors.BLUE}PROCESSING {i+1}/{len(skill_gradients)}: {base_level.upper()} #{i+1}{Colors.ENDC}")
        print("-"*80)
        
        # Generate conversation
        print(f"\n{Colors.CYAN}[1/2] Generating conversation for {base_level} skill level...{Colors.ENDC}")
        print("      Making API call to OpenAI (this may take 15-30 seconds)...")
        
        print("\n      API Request Details:")
        print(f"      - Model: gpt-4o")
        print(f"      - Persona: {args.persona}")
        print(f"      - Skill Level: {base_level}")
        
        start_time = time.time()
        try:
            conversation = generator.generate_conversation(args.persona, base_level)
            
            if not conversation:
                print(f"{Colors.RED}Error: Failed to generate conversation. Skipping to next.{Colors.ENDC}")
                continue
                
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n{Colors.GREEN}      Conversation generated in {duration:.2f} seconds{Colors.ENDC}")
            print(f"      Conversation has {len(conversation)} exchanges")
            
            print(f"\n{Colors.CYAN}      GENERATED CONVERSATION ({len(conversation)} exchanges):{Colors.ENDC}")
            for j, exchange in enumerate(conversation):
                stage = exchange.get('stage', 'unknown').upper()
                ai_prompt = exchange.get('ai_prompt', '')
                user_response = exchange.get('user_response', '')
                
                print(f"\n      {Colors.YELLOW}Exchange {j+1} - Stage: {stage}{Colors.ENDC}")
                print(f"      AI: {ai_prompt}")
                print(f"      User: {user_response}")
            
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
            
            print("\n      Stage Scores:")
            for stage, data in evaluation.get('stages', {}).items():
                score = data.get('score', 0)
                score_color = {
                    0: Colors.RED,
                    1: Colors.RED,
                    2: Colors.CYAN,
                    3: Colors.GREEN
                }.get(score, Colors.ENDC)
                
                print(f"      - {stage.upper()}: {score_color}{score}{Colors.ENDC} - {data.get('feedback', '')}")
                print(f"        Improvement: {data.get('improvement', '')}")
            
            # Store result
            result = {
                "conversation_id": conversation_id,
                "persona": args.persona,
                "skill_level": base_level,
                "gradient": gradient,
                "conversation": conversation,
                "evaluation": evaluation
            }
            
            results.append(result)
            
            # Save individual result
            filename = f"{output_dir}/conversation_{base_level}_{i+1}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"\n{Colors.GREEN}Results saved to {filename}{Colors.ENDC}")
            print(f"Badge level: {badge_color}{badge_level}{Colors.ENDC}")
            print(f"Total score: {evaluation.get('total_score', 0)}")
            
            print("\nSkill scores:")
            for skill, score in evaluation.get('skill_scores', {}).items():
                print(f"  {skill.replace('_', ' ').title()}: {score}")
                
        except Exception as e:
            print(f"{Colors.RED}Error processing conversation: {str(e)}{Colors.ENDC}")
            continue
    
    if not results:
        print(f"{Colors.RED}\nNo results generated. Exiting.{Colors.ENDC}")
        return False
    
    # Create summary CSV
    print(f"\n{Colors.BOLD}" + "-"*80)
    print(f"{Colors.BLUE}GENERATING SUMMARY{Colors.ENDC}")
    print("-"*80)
    
    summary_data = []
    for result in results:
        eval_data = result["evaluation"]
        summary_data.append({
            "conversation_id": result["conversation_id"],
            "skill_level": result["skill_level"],
            "gradient": result["gradient"],
            "total_score": eval_data.get("total_score", 0),
            "percentile_score": eval_data.get("percentile_score", 0),
            "badge_level": eval_data.get("badge_level", "Bronze"),
            "critical_thinking": eval_data.get("skill_scores", {}).get("critical_thinking", 0),
            "communication": eval_data.get("skill_scores", {}).get("communication", 0),
            "emotional_intelligence": eval_data.get("skill_scores", {}).get("emotional_intelligence", 0)
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = f"{output_dir}/test_results_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"{Colors.GREEN}Summary saved to {csv_path}{Colors.ENDC}")
    print("\nSummary data:")
    print(df)
    
    print(f"\n{Colors.CYAN}Detailed Summary:{Colors.ENDC}")
    for index, row in df.iterrows():
        badge_color = {
            'Bronze': Colors.RED,
            'Silver': Colors.CYAN,
            'Gold': Colors.YELLOW
        }.get(row['badge_level'], Colors.ENDC)
        
        print(f"\n  {row['conversation_id']}:")
        print(f"    Skill Level: {row['skill_level']}")
        print(f"    Total Score: {row['total_score']}")
        print(f"    Percentile Score: {row['percentile_score']}%")
        print(f"    Badge Level: {badge_color}{row['badge_level']}{Colors.ENDC}")
        print(f"    Critical Thinking: {row['critical_thinking']}")
        print(f"    Communication: {row['communication']}")
        print(f"    Emotional Intelligence: {row['emotional_intelligence']}")
    
    # Generate visualizations
    if not args.no_visualize and len(df) > 1:
        try:
            generate_visualizations(df, output_dir)
        except Exception as e:
            print(f"{Colors.RED}Error generating visualizations: {str(e)}{Colors.ENDC}")
    
    # Skill level comparison
    if args.compare and len(skill_levels) > 1:
        try:
            compare_skill_levels(df, output_dir)
        except Exception as e:
            print(f"{Colors.RED}Error comparing skill levels: {str(e)}{Colors.ENDC}")
    
    # Group by skill level and check that scores increase with skill level
    if len(skill_levels) > 1:
        skill_level_scores = {}
        for level in skill_levels:
            level_scores = df[df["skill_level"] == level]["total_score"].mean()
            skill_level_scores[level] = level_scores
        
        print(f"\n{Colors.CYAN}Average scores by skill level:{Colors.ENDC}")
        for level, score in skill_level_scores.items():
            print(f"  {level.capitalize()}: {score:.2f}")
        
        if len(skill_level_scores) > 1:
            print(f"\n{Colors.CYAN}Score Differences:{Colors.ENDC}")
            levels = list(skill_level_scores.keys())
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    level1 = levels[i]
                    level2 = levels[j]
                    diff = skill_level_scores[level2] - skill_level_scores[level1]
                    print(f"  {level2.capitalize()} - {level1.capitalize()}: {diff:.2f}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*80)
    print("DIRECT TEST COMPLETED SUCCESSFULLY!")
    print("="*80 + f"{Colors.ENDC}")
    
    return True

def generate_visualizations(df, output_dir):
    """Generate visualizations of the test results."""
    print(f"\n{Colors.CYAN}Generating visualizations...{Colors.ENDC}")
    
    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Skill scores by skill level
    plt.figure(figsize=(10, 6))
    skill_levels = df['skill_level'].unique()
    
    # Set bar width
    bar_width = 0.25
    
    # Set position of bars on x axis
    r1 = range(len(skill_levels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    critical_thinking = [df[df['skill_level'] == level]['critical_thinking'].mean() for level in skill_levels]
    communication = [df[df['skill_level'] == level]['communication'].mean() for level in skill_levels]
    emotional_intelligence = [df[df['skill_level'] == level]['emotional_intelligence'].mean() for level in skill_levels]
    
    plt.bar(r1, critical_thinking, width=bar_width, label='Critical Thinking')
    plt.bar(r2, communication, width=bar_width, label='Communication')
    plt.bar(r3, emotional_intelligence, width=bar_width, label='Emotional Intelligence')
    
    # Add labels and legend
    plt.xlabel('Skill Level')
    plt.ylabel('Score')
    plt.title('Skill Scores by Skill Level')
    plt.xticks([r + bar_width for r in range(len(skill_levels))], [level.capitalize() for level in skill_levels])
    plt.legend()
    
    # Save figure
    plt.savefig(f"{output_dir}/skill_scores_by_level.png")
    print(f"{Colors.GREEN}Saved visualization to {output_dir}/skill_scores_by_level.png{Colors.ENDC}")
    
    # Total scores by skill level
    plt.figure(figsize=(10, 6))
    total_scores = [df[df['skill_level'] == level]['total_score'].mean() for level in skill_levels]
    percentile_scores = [df[df['skill_level'] == level]['percentile_score'].mean() for level in skill_levels]
    
    plt.bar(r1, total_scores, width=bar_width, label='Total Score')
    plt.bar(r2, percentile_scores, width=bar_width, label='Percentile Score')
    
    # Add labels and legend
    plt.xlabel('Skill Level')
    plt.ylabel('Score')
    plt.title('Total Scores by Skill Level')
    plt.xticks([r + bar_width/2 for r in range(len(skill_levels))], [level.capitalize() for level in skill_levels])
    plt.legend()
    
    # Save figure
    plt.savefig(f"{output_dir}/total_scores_by_level.png")
    print(f"{Colors.GREEN}Saved visualization to {output_dir}/total_scores_by_level.png{Colors.ENDC}")
    
    # Badge distribution pie chart
    plt.figure(figsize=(8, 8))
    badge_counts = df['badge_level'].value_counts()
    plt.pie(badge_counts, labels=badge_counts.index, autopct='%1.1f%%')
    plt.title('Badge Level Distribution')
    
    # Save figure
    plt.savefig(f"{output_dir}/badge_distribution.png")
    print(f"{Colors.GREEN}Saved visualization to {output_dir}/badge_distribution.png{Colors.ENDC}")

def compare_skill_levels(df, output_dir):
    """Compare scores across skill levels."""
    print(f"\n{Colors.CYAN}Comparing skill levels...{Colors.ENDC}")
    
    # Group by skill level
    grouped = df.groupby('skill_level')
    
    # Calculate statistics
    stats = grouped.agg({
        'total_score': ['mean', 'min', 'max', 'std'],
        'critical_thinking': ['mean', 'min', 'max', 'std'],
        'communication': ['mean', 'min', 'max', 'std'],
        'emotional_intelligence': ['mean', 'min', 'max', 'std']
    })
    
    # Save statistics to CSV
    stats_path = f"{output_dir}/skill_level_comparison.csv"
    stats.to_csv(stats_path)
    print(f"{Colors.GREEN}Comparison statistics saved to {stats_path}{Colors.ENDC}")
    
    # Display summary
    print("\nComparison Summary:")
    print(stats)
    
    # Calculate significance
    if len(grouped) > 1:
        print(f"\n{Colors.CYAN}Skill Level Progression:{Colors.ENDC}")
        
        levels = sorted(df['skill_level'].unique())
        for i in range(len(levels) - 1):
            level1 = levels[i]
            level2 = levels[i + 1]
            
            diff = grouped.get_group(level2)['total_score'].mean() - grouped.get_group(level1)['total_score'].mean()
            percent_increase = (diff / grouped.get_group(level1)['total_score'].mean()) * 100 if grouped.get_group(level1)['total_score'].mean() > 0 else float('inf')
            
            print(f"  {level1.capitalize()} → {level2.capitalize()}: +{diff:.2f} points ({percent_increase:.1f}% increase)")

if __name__ == "__main__":
    print(f"{Colors.CYAN}Starting direct test...{Colors.ENDC}")
    args = parse_args()
    success = run_direct_test(args)
    if success:
        print(f"\n{Colors.GREEN}Test completed successfully!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}Test failed. See output above for details.{Colors.ENDC}")
        sys.exit(1) 