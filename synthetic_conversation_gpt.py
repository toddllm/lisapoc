#!/usr/bin/env python3
"""
Synthetic Conversation Generator and Evaluator

This script uses GPT-4o to generate and evaluate synthetic networking conversations
based on the evaluation framework. It creates conversations at varying skill levels
and evaluates them against the criteria for critical thinking, communication, and
emotional intelligence.
"""

import os
import json
import argparse
import random
import requests
from datetime import datetime
from typing import Dict, List, Any, Tuple
import openai
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Set up OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")

# Conversation stages
STAGES = ["opener", "carry", "linkedin", "moveon", "farewell"]

# Personas for the AI conversation partner
PERSONAS = {
    "jake": {
        "name": "Jake",
        "role": "Software Engineer",
        "company": "TechCorp",
        "industry": "tech",
        "specialty": "AI systems",
        "years": "5"
    },
    "sarah": {
        "name": "Sarah",
        "role": "Marketing Director",
        "company": "Innovate Inc",
        "industry": "marketing",
        "specialty": "digital campaigns",
        "years": "8"
    }
}

# Skill levels for user responses - updated with clearer distinctions
SKILL_LEVELS = {
    "novice": "The user is new to networking and struggles with social interactions. They tend to use closed questions, talk about themselves too much, and have difficulty transitioning between topics smoothly. They make abrupt statements, use poor networking etiquette, and often fail to demonstrate active listening or genuine interest in others. Their responses are typically short, lack substance, and focus on their own needs rather than building relationships.",
    
    "intermediate": "The user has some networking experience but is still developing their skills. They can maintain a conversation and ask some open-ended questions, but may not always read social cues well or transition smoothly between topics. Their LinkedIn connection requests may feel somewhat forced. They show moderate critical thinking and emotional intelligence but sometimes miss opportunities to deepen the conversation. They generally demonstrate good manners but occasionally lapse into self-focus.",
    
    "advanced": "The user is skilled at networking. They ask thoughtful, open-ended questions that demonstrate genuine interest and active listening. They make insightful connections between topics, share relevant personal experiences, and transition smoothly between conversation stages. Their connection requests feel natural and mutually beneficial. They demonstrate high emotional intelligence by reading social cues correctly and adapting their approach appropriately. They consistently focus on relationship-building and leave a positive, memorable impression."
}

# Success criteria for each skill area
SUCCESS_CRITERIA = {
    "critical_thinking": [
        "Identify and adapt to the other person's level of engagement",
        "Make logical, relevant connections between your work and theirs",
        "Anticipate possible responses and adjust your approach accordingly",
        "Recognize when a conversation isn't progressing and pivot strategically",
        "Ask insightful, open-ended questions that encourage deeper discussion",
        "Use the most effective tactic/language to reach your goals"
    ],
    "communication": [
        "Express your thoughts clearly and concisely without over-explaining",
        "Use approachable, engaging language that fits the context",
        "Make your ask (e.g., LinkedIn connection) feel natural, not transactional",
        "Ensure a smooth flow in conversation, avoiding abrupt topic shifts",
        "Balance talking and listening, keeping the conversation dynamic",
        "Ask questions of the other person to go deeper on what they said"
    ],
    "emotional_intelligence": [
        "Read verbal and nonverbal cues to gauge interest and engagement",
        "Mirror energy and tone to create a comfortable interaction",
        "Adjust your approach based on the other person's mood or behavior",
        "Show genuine curiosity and interest in their work, not just your goals",
        "Know when to exit a conversation gracefully without making it awkward",
        "Be polite",
        "Talk about topics which are non-controversial",
        "Be open-minded",
        "Talk about topics which most people are familiar with"
    ]
}

# Example responses for each stage and skill level
EXAMPLE_RESPONSES = {
    "opener": {
        "novice": [
            "Hey.",
            "Nice weather, huh?",
            "I hate these things."
        ],
        "intermediate": [
            "What do you think of the food?",
            "You look familiar. Do I know you from somewhere?",
            "How's the event going for you so far?"
        ],
        "advanced": [
            "What brings you here today?",
            "First time at one of these events?",
            "What do you think of the event?",
            "I'm new to this event. Any tips on making the most of it?"
        ]
    },
    "carry": {
        "novice": [
            "Oh.",
            "Cool.",
            "Anyway, let me tell you about myself..."
        ],
        "intermediate": [
            "That sounds interesting.",
            "I've heard of that company.",
            "How long have you been doing that?"
        ],
        "advanced": [
            "What do you do?",
            "What got you started in that?",
            "Tell me more about that."
        ]
    },
    "linkedin": {
        "novice": [
            "Give me your LinkedIn.",
            "You should follow me on LinkedIn.",
            "I need more connections on LinkedIn."
        ],
        "intermediate": [
            "Are you on LinkedIn?",
            "Should we exchange contact info?"
        ],
        "advanced": [
            "Why don't we connect on LinkedIn to keep in touch?",
            "I'd love to exchange insights. Would you like to connect on LinkedIn?"
        ]
    },
    "moveon": {
        "novice": [
            "I'm bored. I'm going to talk to someone else.",
            "Gotta go.",
            "I see someone more important."
        ],
        "intermediate": [
            "I should probably mingle a bit more.",
            "I think I need to say hello to a few other people."
        ],
        "advanced": [
            "I see someone over there I've been wanting to talk to.",
            "Would you excuse me?"
        ]
    },
    "farewell": {
        "novice": [
            "Bye.",
            "See ya.",
            ""  # Just walking away
        ],
        "intermediate": [
            "Thanks for chatting.",
            "Have a good rest of the event."
        ],
        "advanced": [
            "It's been great talking to you.",
            "It was nice meeting you.",
            "I enjoyed our conversation."
        ]
    }
}

# Define numerical weights for how stages contribute to skill areas
SKILL_AREA_WEIGHTS = {
    "opener": {
        "critical_thinking": 0.35,
        "communication": 0.35,
        "emotional_intelligence": 0.30
    },
    "carry": {
        "critical_thinking": 0.35,
        "communication": 0.35,
        "emotional_intelligence": 0.30
    },
    "linkedin": {
        "critical_thinking": 0.40,
        "communication": 0.35,
        "emotional_intelligence": 0.25
    },
    "moveon": {
        "critical_thinking": 0.30,
        "communication": 0.30,
        "emotional_intelligence": 0.40
    },
    "farewell": {
        "critical_thinking": 0.25,
        "communication": 0.35,
        "emotional_intelligence": 0.40
    }
}

# Badge level thresholds - recalibrated for better distribution
BADGE_THRESHOLDS = {
    "Bronze": {
        "min_skill_score": 0,    # For very low scores
        "total_score_range": (0, 7)  # Adjusted for clearer separation
    },
    "Silver": {
        "min_skill_score": 2,    # Moderate skill level
        "total_score_range": (8, 14)  # Middle range
    },
    "Gold": {
        "min_skill_score": 4,    # High skill level
        "total_score_range": (15, 24)  # Upper range for truly exceptional performance
    }
}

class ConversationGenerator:
    """Generates synthetic networking conversations using GPT-4o."""
    
    def __init__(self):
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    def generate_conversation(self, persona_id: str, skill_level: str) -> List[Dict[str, Any]]:
        """
        Generate a complete synthetic conversation between a user and AI persona.
        
        Args:
            persona_id: ID of the AI persona
            skill_level: Skill level of the user (novice, intermediate, advanced)
            
        Returns:
            List of conversation exchanges
        """
        persona = PERSONAS.get(persona_id, PERSONAS["jake"])
        skill_description = SKILL_LEVELS.get(skill_level, SKILL_LEVELS["intermediate"])
        
        # Create the initial system prompt with enhanced instructions
        system_prompt = f"""
        You are simulating a networking conversation at a professional event.

        You will play two roles:
        1. {persona['name']}, a {persona['role']} at {persona['company']} who specializes in {persona['specialty']} with {persona['years']} years of experience
        2. A person attending the networking event who has the following skill level: {skill_description}

        Generate a realistic conversation between these two people that includes all five stages:
        1. Opener: Initial greeting and conversation starter
        2. Carry: Maintaining dialogue flow and asking about the other person
        3. LinkedIn: Asking to connect professionally
        4. Moveon: Gracefully transitioning away from the conversation
        5. Farewell: Closing the conversation politely

        IMPORTANT: Use EXACTLY these stage names in your JSON output: "opener", "carry", "linkedin", "moveon", "farewell"

        For each stage, generate both {persona['name']}'s response and the networking person's response.
        The networking person's responses should VERY CLEARLY reflect their skill level ({skill_level}).

        If the skill level is "novice":
        - Use simple, sometimes awkward language with short responses
        - Include mostly closed-ended questions or no questions at all
        - Focus mainly on themselves rather than showing interest in others
        - Use abrupt transitions between topics
        - Include statements like "Um," "Uh," "I guess," "Whatever," "Sure"
        - Make the LinkedIn connection request feel transactional and self-serving
        - Create a farewell that feels abrupt and lacks warmth

        If the skill level is "intermediate":
        - Use moderately engaging language and decent social etiquette
        - Mix open and closed-ended questions
        - Show some interest in the other person but miss opportunities
        - Include some industry knowledge but not deeply specific
        - Transition between topics with reasonable smoothness
        - Make the LinkedIn connection somewhat natural but not exceptional
        - Create a polite but somewhat generic farewell

        If the skill level is "advanced":
        - Use sophisticated, engaging language with excellent social etiquette
        - Ask primarily thoughtful, open-ended questions
        - Demonstrate active listening by referring back to previous points
        - Show detailed industry knowledge with specific terminology
        - Transition between topics naturally and skillfully
        - Make the LinkedIn connection feel mutual and value-focused
        - Create a warm, memorable farewell that reinforces the connection

        Format your response as a JSON object with a "conversation" field containing an array of exchanges, where each exchange has:
        - "stage": The conversation stage (must be one of: "opener", "carry", "linkedin", "moveon", "farewell")
        - "ai_prompt": What {persona['name']} says
        - "user_response": What the networking person says

        Here are examples of {skill_level} level responses for each stage:
        {json.dumps({stage: EXAMPLE_RESPONSES[stage][skill_level] for stage in STAGES}, indent=2)}

        For variety, try to include 5-7 exchanges in total. MAKE SURE the skill level differences are EXTREMELY CLEAR and DISTINCT. The novice should be clearly struggling, the intermediate somewhat effective, and the advanced truly exceptional.
        """
        
        # Generate the conversation using direct API call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a realistic networking conversation between {persona['name']} and a {skill_level} networker. Make sure the {skill_level} skill level is VERY CLEAR in their responses."}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Parse the response
            conversation_data = json.loads(response_data["choices"][0]["message"]["content"])
            return conversation_data.get("conversation", [])
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return []

class ConversationEvaluator:
    """Evaluates networking conversations using GPT-4o."""
    
    def __init__(self):
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    def evaluate_response(self, stage: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a single user response using GPT-4o.
        
        Args:
            stage: The conversation stage (opener, carry, etc.)
            response: The user's response text
            
        Returns:
            Evaluation results with score and feedback
        """
        # Normalize stage name to handle variations
        normalized_stage = stage.lower()
        if normalized_stage == "carrying_conversation" or normalized_stage == "carrying":
            normalized_stage = "carry"
        elif normalized_stage == "linkedin_connection":
            normalized_stage = "linkedin"
        elif normalized_stage == "moving_on":
            normalized_stage = "moveon"
        
        # Make sure the normalized stage exists in our examples
        if normalized_stage not in STAGES:
            print(f"Warning: Unknown stage '{stage}' normalized to '{normalized_stage}', falling back to 'carry'")
            normalized_stage = "carry"  # Default fallback
        
        system_prompt = f"""
        You are an expert evaluator of networking skills. Your task is to evaluate a response given during the "{stage}" stage of a networking conversation.

        Evaluate the response on a scale of 0-3:
        - 3 points: Optimal response that demonstrates excellent networking skills
        - 2 points: Good response that is effective but could be improved
        - 1 point: Basic response that needs significant improvement
        - 0 points: Poor response that demonstrates ineffective networking

        For reference, here are examples of responses at different skill levels for this stage:
        - Advanced (3 points): {", ".join(EXAMPLE_RESPONSES[normalized_stage]["advanced"][:2])}
        - Intermediate (2 points): {", ".join(EXAMPLE_RESPONSES[normalized_stage]["intermediate"][:2])}
        - Novice (0-1 points): {", ".join(EXAMPLE_RESPONSES[normalized_stage]["novice"][:2])}

        Also evaluate how this response demonstrates skills in three areas:
        1. Critical Thinking (0-3 points): 
           - Does the response show analysis of the situation?
           - Does it make logical connections?
           - Does it adapt to the context appropriately?
           - Does it demonstrate strategic question choice?

        2. Communication (0-3 points):
           - Is the language clear and appropriate?
           - Does it maintain good conversation flow?
           - Does it balance speaking and listening effectively?
           - Does it demonstrate proper networking etiquette?

        3. Emotional Intelligence (0-3 points):
           - Does it show awareness of social cues?
           - Does it demonstrate genuine interest?
           - Is it appropriately polite and considerate?
           - Does it show empathy and relationship-building skills?

        Be extremely strict and deliberate in your evaluation. A score of 3 should only be given for truly exceptional responses that demonstrate mastery.
        A score of 2 should be given for solid responses with minor issues. 
        A score of 1 should be given for basic responses with significant room for improvement.
        A score of 0 should be given for poor responses that fail to demonstrate the skill.

        Provide your evaluation as a JSON object with:
        - score: Numeric score (0-3)
        - feedback: Brief explanation of the score
        - improvement: Suggestion for improvement
        - skill_scores: Scores for each skill area (critical_thinking, communication, emotional_intelligence) with each on a scale of 0-3
        - skill_feedback: Brief feedback for each skill area
        """
        
        response_prompt = f"""
        Evaluate this networking response during the "{stage}" stage of a conversation:

        "{response}"

        Remember to be strict in your evaluation. Only give high scores (2-3) to responses that truly demonstrate skill.
        """
        
        # Generate the evaluation using direct API call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2  # Lowered temperature for more consistent evaluations
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Parse the response
            evaluation = json.loads(response_data["choices"][0]["message"]["content"])
            return evaluation
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "score": 0,
                "feedback": "Error evaluating response",
                "improvement": "N/A",
                "skill_scores": {
                    "critical_thinking": 0,
                    "communication": 0,
                    "emotional_intelligence": 0
                },
                "skill_feedback": {
                    "critical_thinking": "Error",
                    "communication": "Error",
                    "emotional_intelligence": "Error"
                }
            }
    
    def evaluate_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a complete conversation.
        
        Args:
            conversation: List of conversation exchanges
            
        Returns:
            Evaluation results with scores and feedback
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "skill_scores": {
                "critical_thinking": 0,
                "communication": 0,
                "emotional_intelligence": 0
            },
            "total_score": 0,
            "badges": {},
            "detailed_feedback": []  # Added for more comprehensive feedback
        }
        
        # Track skill scores directly from evaluations
        direct_skill_scores = {
            "critical_thinking": [],
            "communication": [],
            "emotional_intelligence": []
        }
        
        # Evaluate each stage
        for exchange in conversation:
            stage = exchange.get("stage")
            user_response = exchange.get("user_response")
            
            # Skip if missing data
            if not stage or not user_response:
                continue
                
            # Evaluate the response
            evaluation = self.evaluate_response(stage, user_response)
            
            # Store the evaluation in detailed feedback
            results["detailed_feedback"].append({
                "stage": stage,
                "response": user_response,
                "evaluation": evaluation
            })
            
            # Store the results
            results["stages"][stage] = {
                "score": evaluation.get("score", 0),
                "feedback": evaluation.get("feedback", ""),
                "improvement": evaluation.get("improvement", ""),
                "user_response": user_response,
                "skill_scores": evaluation.get("skill_scores", {})  # Individual skill scores for the stage
            }
            
            # Add to total score
            stage_score = evaluation.get("score", 0)
            results["total_score"] += stage_score
            
            # Collect direct skill scores from this evaluation
            skill_scores = evaluation.get("skill_scores", {})
            for skill in direct_skill_scores:
                if skill in skill_scores:
                    direct_skill_scores[skill].append(skill_scores[skill])
        
        # Calculate the average skill scores directly from evaluations
        for skill in direct_skill_scores:
            scores = direct_skill_scores[skill]
            if scores:  # Avoid division by zero
                # Calculate the average and scale to match badge thresholds
                avg_score = sum(scores) / len(scores)
                # Scale from 0-3 to 0-5 scale for visualization purposes
                results["skill_scores"][skill] = round(avg_score * 5/3, 1)
        
        # Determine badge level based on total score
        total_score = results["total_score"]
        
        if total_score >= BADGE_THRESHOLDS["Gold"]["total_score_range"][0]:
            badge_level = "Gold"
        elif total_score >= BADGE_THRESHOLDS["Silver"]["total_score_range"][0]:
            badge_level = "Silver"
        else:
            badge_level = "Bronze"
        
        results["badge_level"] = badge_level
        
        # Individual skill badges based on averaged skill scores
        for skill, score in results["skill_scores"].items():
            # Convert back to 0-3 scale for threshold comparison
            scaled_score = score * 3/5
            
            if scaled_score >= 2.5:  # High bar for Gold (was BADGE_THRESHOLDS["Gold"]["min_skill_score"])
                results["badges"][skill] = "Gold"
            elif scaled_score >= 1.5:  # Moderate bar for Silver (was BADGE_THRESHOLDS["Silver"]["min_skill_score"])
                results["badges"][skill] = "Silver"
            else:
                results["badges"][skill] = "Bronze"
        
        # Calculate percentile score 
        max_possible_score = len(conversation) * 3  # 3 points max per exchange
        if max_possible_score > 0:
            results["percentile_score"] = round((results["total_score"] / max_possible_score) * 100, 1)
        else:
            results["percentile_score"] = 0
        
        return results

def generate_and_evaluate(args):
    """Generate and evaluate synthetic conversations."""
    generator = ConversationGenerator()
    evaluator = ConversationEvaluator()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    results = []
    
    # Generate conversations for each skill level
    for skill_level in args.skill_levels:
        print(f"\nGenerating {args.count} {skill_level} level conversations with {args.persona}...")
        
        for i in tqdm(range(args.count)):
            # Generate conversation
            conversation = generator.generate_conversation(args.persona, skill_level)
            
            if not conversation:
                print(f"Failed to generate conversation {i+1}")
                continue
            
            # Display conversation if verbose
            if args.verbose:
                print("\n=== SYNTHETIC CONVERSATION ===")
                for exchange in conversation:
                    print(f"\nStage: {exchange.get('stage', 'unknown').upper()}")
                    print(f"AI: {exchange.get('ai_prompt', '')}")
                    print(f"User: {exchange.get('user_response', '')}")
            
            # Evaluate conversation
            evaluation = evaluator.evaluate_conversation(conversation)
            
            # Store results
            result = {
                "conversation_id": f"{args.persona}_{skill_level}_{i+1}",
                "persona": args.persona,
                "skill_level": skill_level,
                "conversation": conversation,
                "evaluation": evaluation
            }
            
            results.append(result)
            
            # Save individual result
            filename = f"{args.output}/conversation_{args.persona}_{skill_level}_{i+1}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            
            # Display evaluation summary if verbose
            if args.verbose:
                print("\n=== EVALUATION RESULTS ===")
                print(f"Total Score: {evaluation['total_score']}/24")
                print(f"Badge Level: {evaluation['badge_level']}")
                print("\nSkill Scores:")
                for skill, score in evaluation['skill_scores'].items():
                    print(f"  {skill.replace('_', ' ').title()}: {score} - {evaluation['badges'][skill]}")
                print("\nBadge Thresholds:")
                for level, thresholds in BADGE_THRESHOLDS.items():
                    print(f"  {level}: Min Skill Score {thresholds['min_skill_score']}, " 
                          f"Total Score Range {thresholds['total_score_range']}")
    
    # Save all results
    with open(f"{args.output}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_report(results, args.output)
    
    return results

def generate_report(results, output_dir):
    """Generate a summary report of evaluation results."""
    # Extract data for analysis
    data = []
    for result in results:
        data.append({
            "conversation_id": result["conversation_id"],
            "persona": result["persona"],
            "skill_level": result["skill_level"],
            "total_score": result["evaluation"]["total_score"],
            "badge_level": result["evaluation"]["badge_level"],
            "critical_thinking": result["evaluation"]["skill_scores"]["critical_thinking"],
            "communication": result["evaluation"]["skill_scores"]["communication"],
            "emotional_intelligence": result["evaluation"]["skill_scores"]["emotional_intelligence"]
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate summary statistics
    summary = df.groupby("skill_level").agg({
        "total_score": ["mean", "std", "min", "max"],
        "critical_thinking": ["mean", "std"],
        "communication": ["mean", "std"],
        "emotional_intelligence": ["mean", "std"]
    })
    
    # Save summary to CSV
    summary.to_csv(f"{output_dir}/summary_statistics.csv")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Box plot of total scores by skill level
    plt.subplot(2, 2, 1)
    df.boxplot(column="total_score", by="skill_level")
    plt.title("Total Score Distribution by Skill Level")
    plt.suptitle("")
    
    # Bar chart of average skill scores by skill level
    plt.subplot(2, 2, 2)
    skill_means = df.groupby("skill_level")[["critical_thinking", "communication", "emotional_intelligence"]].mean()
    skill_means.plot(kind="bar", ax=plt.gca())
    plt.title("Average Skill Scores by Skill Level")
    plt.ylabel("Score (0-5)")
    
    # Pie chart of badge distribution
    plt.subplot(2, 2, 3)
    badge_counts = df["badge_level"].value_counts()
    plt.pie(badge_counts, labels=badge_counts.index, autopct="%1.1f%%")
    plt.title("Badge Level Distribution")
    
    # Scatter plot of critical thinking vs emotional intelligence
    plt.subplot(2, 2, 4)
    for level in df["skill_level"].unique():
        subset = df[df["skill_level"] == level]
        plt.scatter(subset["critical_thinking"], subset["emotional_intelligence"], 
                   label=level, alpha=0.7)
    plt.xlabel("Critical Thinking Score")
    plt.ylabel("Emotional Intelligence Score")
    plt.title("Critical Thinking vs Emotional Intelligence")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_summary.png")
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Synthetic Conversation Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ margin-bottom: 30px; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            .badge-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Synthetic Conversation Evaluation Report</h1>
        
        <div class="badge-info">
            <h2>Badge Level Thresholds</h2>
            <table>
                <tr>
                    <th>Badge Level</th>
                    <th>Minimum Skill Score</th>
                    <th>Total Score Range</th>
                </tr>
    """
    
    for level, thresholds in BADGE_THRESHOLDS.items():
        min_score = thresholds["min_skill_score"]
        total_min, total_max = thresholds["total_score_range"]
        html_report += f"""
                <tr>
                    <td>{level}</td>
                    <td>{min_score}</td>
                    <td>{total_min} - {total_max}</td>
                </tr>
        """
    
    html_report += """
            </table>
            <p><strong>Note:</strong> To achieve a badge level, a user must meet the minimum score in <em>all</em> skill areas (Critical Thinking, Communication, Emotional Intelligence) and have a total score within the specified range.</p>
        </div>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Skill Level</th>
                    <th>Count</th>
                    <th>Avg Total Score</th>
                    <th>Avg Critical Thinking</th>
                    <th>Avg Communication</th>
                    <th>Avg Emotional Intelligence</th>
                </tr>
    """
    
    for level in df["skill_level"].unique():
        subset = df[df["skill_level"] == level]
        html_report += f"""
                <tr>
                    <td>{level}</td>
                    <td>{len(subset)}</td>
                    <td>{subset["total_score"].mean():.2f}</td>
                    <td>{subset["critical_thinking"].mean():.2f}</td>
                    <td>{subset["communication"].mean():.2f}</td>
                    <td>{subset["emotional_intelligence"].mean():.2f}</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="visualization">
            <h2>Visualizations</h2>
            <img src="evaluation_summary.png" alt="Evaluation Summary" width="800">
        </div>
        
        <div class="conversations">
            <h2>Individual Conversations</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Skill Level</th>
                    <th>Total Score</th>
                    <th>Badge</th>
                    <th>Critical Thinking</th>
                    <th>Communication</th>
                    <th>Emotional Intelligence</th>
                    <th>Link</th>
                </tr>
    """
    
    for _, row in df.iterrows():
        html_report += f"""
                <tr>
                    <td>{row["conversation_id"]}</td>
                    <td>{row["skill_level"]}</td>
                    <td>{row["total_score"]}</td>
                    <td>{row["badge_level"]}</td>
                    <td>{row["critical_thinking"]:.1f}</td>
                    <td>{row["communication"]:.1f}</td>
                    <td>{row["emotional_intelligence"]:.1f}</td>
                    <td><a href="conversation_{row['conversation_id']}.json">View Details</a></td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/report.html", "w") as f:
        f.write(html_report)
    
    print(f"\nReport generated at {output_dir}/report.html")

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate synthetic networking conversations")
    parser.add_argument("--persona", choices=list(PERSONAS.keys()), default="jake", 
                        help="Persona for the conversation partner")
    parser.add_argument("--skill-levels", nargs="+", choices=list(SKILL_LEVELS.keys()), default=["novice", "intermediate", "advanced"],
                        help="Skill levels to generate")
    parser.add_argument("--count", type=int, default=3, 
                        help="Number of conversations to generate per skill level")
    parser.add_argument("--output", default="results", 
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", 
                        help="Display detailed output")
    
    args = parser.parse_args()
    
    print(f"Synthetic Conversation Generator and Evaluator")
    print(f"============================================")
    print(f"Using OpenAI API with model: gpt-4o")
    print(f"Generating {args.count} conversations per skill level")
    print(f"Skill levels: {', '.join(args.skill_levels)}")
    print(f"Persona: {args.persona}")
    print(f"Output directory: {args.output}")
    print(f"Badge thresholds:")
    for level, thresholds in BADGE_THRESHOLDS.items():
        print(f"  {level}: Min Skill Score {thresholds['min_skill_score']}, " 
              f"Total Score Range {thresholds['total_score_range']}")
    
    generate_and_evaluate(args)

if __name__ == "__main__":
    main() 