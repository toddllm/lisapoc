#!/usr/bin/env python3
"""
Synthetic Conversation Tester

This script generates synthetic conversations between a user and AI persona,
then evaluates them based on the scoring framework to test the evaluation engine.
"""

import json
import random
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Define scoring criteria based on the framework
SCORING_CRITERIA = {
    "opener": {
        "optimal": [
            "What brings you here today/tonight?",
            "First time at one of these events?",
            "What do you think of the event?",
            "I'm new to this event. Any tips on making the most of it?"
        ],
        "good": [
            "What do you think of the food?",
            "You look familiar. Do I know you from somewhere?",
            "How's the event going for you so far?"
        ],
        "poor": [
            "Hey.",
            "Nice weather, huh?",
            "I hate these things."
        ]
    },
    "carry": {
        "optimal": [
            "What do you do?",
            "What got you started in that?",
            "Tell me more about that."
        ],
        "good": [
            "That sounds interesting.",
            "I've heard of that company.",
            "How long have you been doing that?"
        ],
        "poor": [
            "Oh.",
            "Cool.",
            "Anyway, let me tell you about myself..."
        ]
    },
    "linkedin": {
        "optimal": [
            "Why don't we connect on LinkedIn to keep in touch?",
            "I'd love to exchange insights. Would you like to connect on LinkedIn?"
        ],
        "good": [
            "Are you on LinkedIn?",
            "Should we exchange contact info?"
        ],
        "poor": [
            "Give me your LinkedIn.",
            "You should follow me on LinkedIn.",
            "I need more connections on LinkedIn."
        ]
    },
    "moveon": {
        "optimal": [
            "I see someone over there I've been wanting to talk to.",
            "Would you excuse me?"
        ],
        "good": [
            "I should probably mingle a bit more.",
            "I think I need to say hello to a few other people."
        ],
        "poor": [
            "I'm bored. I'm going to talk to someone else.",
            "Gotta go.",
            "I see someone more important."
        ]
    },
    "farewell": {
        "optimal": [
            "It's been great talking to you.",
            "It was nice meeting you.",
            "I enjoyed our conversation."
        ],
        "good": [
            "Thanks for chatting.",
            "Have a good rest of the event."
        ],
        "poor": [
            "Bye.",
            "See ya.",
            ""  # Just walking away
        ]
    }
}

# AI persona responses to keep conversations flowing
AI_RESPONSES = {
    "opener": [
        "Hi there! I'm {name}. Nice to meet you!",
        "Hello! I'm {name}. I work in {role}.",
        "Hey! I'm {name}. First time at this event?"
    ],
    "carry": [
        "I work as a {role} at {company}. I focus on {specialty}.",
        "I've been in {industry} for about {years} years, mainly working on {specialty}.",
        "Currently I'm at {company} as a {role}. What about you?"
    ],
    "linkedin": [
        "Sure, I'd be happy to connect on LinkedIn!",
        "That sounds great, I'm always looking to expand my professional network.",
        "Absolutely, let me get my phone out."
    ],
    "moveon": [
        "No problem at all, it was nice talking with you.",
        "Of course, I should probably mingle a bit more too.",
        "Sure thing, thanks for chatting."
    ],
    "farewell": [
        "It was great meeting you too! Hope to see you at the next event.",
        "Likewise! Have a great rest of your day.",
        "Nice meeting you as well! Enjoy the rest of the event."
    ]
}

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

class EvaluationEngine:
    """Evaluates conversations based on the scoring framework."""
    
    def evaluate_response(self, stage: str, response: str) -> Tuple[int, str]:
        """
        Evaluate a single user response.
        
        Args:
            stage: The conversation stage (opener, carry, etc.)
            response: The user's response text
            
        Returns:
            Tuple of (score, feedback)
        """
        criteria = SCORING_CRITERIA.get(stage, {})
        
        # Simple exact matching for this prototype
        if response in criteria.get("optimal", []):
            return 3, f"Excellent {stage} response. You used optimal language."
        elif response in criteria.get("good", []):
            return 2, f"Good {stage} response. Consider using more open-ended questions."
        else:
            # Check for partial matches with optimal phrases
            for phrase in criteria.get("optimal", []):
                if phrase.lower() in response.lower():
                    return 2, f"Good attempt at {stage}. Your phrasing could be more direct."
            
            # Check for partial matches with good phrases    
            for phrase in criteria.get("good", []):
                if phrase.lower() in response.lower():
                    return 1, f"Your {stage} has the right idea but needs more polishing."
                    
            return 0, f"This {stage} response could be improved. Try using more engaging language."
    
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
            "badges": {}
        }
        
        for exchange in conversation:
            stage = exchange.get("stage")
            response = exchange.get("user_response")
            
            score, feedback = self.evaluate_response(stage, response)
            
            results["stages"][stage] = {
                "score": score,
                "feedback": feedback,
                "max_possible": 3,
                "user_response": response
            }
            
            results["total_score"] += score
            
            # Assign points to skills based on the stage
            # This is a simplified allocation - would be more sophisticated in production
            if stage in ["opener", "linkedin"]:
                results["skill_scores"]["critical_thinking"] += score
            if stage in ["carry", "farewell"]:
                results["skill_scores"]["communication"] += score
            if stage in ["moveon", "opener"]:
                results["skill_scores"]["emotional_intelligence"] += score
        
        # Determine badge levels
        for skill, score in results["skill_scores"].items():
            if score >= 5:
                results["badges"][skill] = "Gold"
            elif score >= 3:
                results["badges"][skill] = "Silver"
            else:
                results["badges"][skill] = "Bronze"
        
        return results

def generate_synthetic_conversation(persona_id: str, quality_level: str = "mixed") -> List[Dict[str, Any]]:
    """
    Generate a synthetic conversation between user and AI persona.
    
    Args:
        persona_id: ID of the AI persona
        quality_level: Quality of user responses (optimal, good, poor, or mixed)
        
    Returns:
        List of conversation exchanges
    """
    persona = PERSONAS.get(persona_id, PERSONAS["jake"])
    conversation = []
    
    for stage in ["opener", "carry", "linkedin", "moveon", "farewell"]:
        # Select AI response and format with persona details
        ai_response = random.choice(AI_RESPONSES[stage])
        ai_response = ai_response.format(**persona)
        
        # Select user response based on quality level
        if quality_level == "optimal":
            user_response = random.choice(SCORING_CRITERIA[stage]["optimal"])
        elif quality_level == "good":
            user_response = random.choice(SCORING_CRITERIA[stage]["good"])
        elif quality_level == "poor":
            user_response = random.choice(SCORING_CRITERIA[stage]["poor"])
        else:  # mixed
            quality = random.choice(["optimal", "good", "poor"])
            user_response = random.choice(SCORING_CRITERIA[stage][quality])
        
        conversation.append({
            "stage": stage,
            "ai_prompt": ai_response,
            "user_response": user_response
        })
    
    return conversation

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate synthetic conversations")
    parser.add_argument("--persona", choices=list(PERSONAS.keys()), default="jake", 
                        help="Persona for the conversation partner")
    parser.add_argument("--quality", choices=["optimal", "good", "poor", "mixed"], default="mixed",
                        help="Quality level of user responses")
    parser.add_argument("--count", type=int, default=1, help="Number of conversations to generate")
    parser.add_argument("--output", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    evaluator = EvaluationEngine()
    
    for i in range(args.count):
        print(f"Generating conversation {i+1}/{args.count}...")
        conversation = generate_synthetic_conversation(args.persona, args.quality)
        
        # Display conversation
        print("\n=== SYNTHETIC CONVERSATION ===")
        for exchange in conversation:
            print(f"\nStage: {exchange['stage'].upper()}")
            print(f"AI: {exchange['ai_prompt']}")
            print(f"User: {exchange['user_response']}")
        
        # Evaluate conversation
        evaluation = evaluator.evaluate_conversation(conversation)
        
        # Display evaluation results
        print("\n=== EVALUATION RESULTS ===")
        print(f"Total Score: {evaluation['total_score']}/15")
        print("\nSkill Scores:")
        for skill, score in evaluation['skill_scores'].items():
            print(f"  {skill.replace('_', ' ').title()}: {score}")
        
        print("\nBadges Earned:")
        for skill, badge in evaluation['badges'].items():
            print(f"  {skill.replace('_', ' ').title()}: {badge}")
        
        print("\nDetailed Feedback:")
        for stage, details in evaluation['stages'].items():
            print(f"  {stage.upper()}: {details['score']}/3 - {details['feedback']}")
        
        # Save results to file
        result = {
            "conversation": conversation,
            "evaluation": evaluation
        }
        
        filename = f"{args.output}/conversation_{args.persona}_{args.quality}_{i+1}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main() 