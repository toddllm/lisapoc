# Conversation Evaluation System Design

## Overview

This document outlines the design for a comprehensive conversation evaluation system that scores user interactions based on their networking skills. The system will evaluate conversations across three key skill areas and provide dynamic badges based on performance.

## Evaluation Framework

### Scoring Dimensions

1. **Critical Thinking (Tactics)** - How effectively users deploy conversational strategies
2. **Communication (Language Used)** - Quality of expression and dialogue flow
3. **Emotional Intelligence** - Ability to read and respond to social cues appropriately

### Conversation Stages & Scoring (15 points total)

| Stage | Description | Point Values | Example Responses |
|-------|-------------|--------------|-------------------|
| Opener | Initial conversation starter | 3 points (optimal), 2 points (good), 0-1 points (needs improvement) | **3-point**: "What brings you here today/tonight?", "First time at one of these events?", "What do you think of the event?", "I'm new to this event. Any tips on making the most of it?" <br> **2-point**: "What do you think of the food?", "You look familiar. Do I know you from somewhere?", "How's the event going for you so far?" |
| Carrying Conversation | Maintaining dialogue flow | 3 points (optimal), 2 points (good), 0-1 points (needs improvement) | **3-point**: "What do you do?", "What got you started in that?", "Tell me more about that." <br> **2-point**: "That sounds interesting", "How long have you been doing that?" |
| LinkedIn Connection | Asking for professional connection | 3 points (optimal), 2 points (good), 0-1 points (needs improvement) | **3-point**: "Why don't we connect on LinkedIn to keep in touch?", "I'd love to exchange insights. Would you like to connect on LinkedIn?" <br> **2-point**: "Are you on LinkedIn?" |
| Move On | Gracefully transitioning away | 3 points (optimal), 2 points (good), 0-1 points (needs improvement) | **3-point**: "I see someone over there I've been wanting to talk to", "Would you excuse me?" <br> **2-point**: "I should probably mingle a bit more" |
| Farewell | Closing the conversation | 3 points (optimal), 2 points (good), 0-1 points (needs improvement) | **3-point**: "It's been great/nice talking to you", "It was nice meeting you" <br> **2-point**: "Have a good rest of the event" |

### Skill Area Distribution (Numerical Weights)

Each stage contributes points to different skill areas based on the following numerical weights:

| Stage | Critical Thinking | Communication | Emotional Intelligence |
|-------|-------------------|---------------|------------------------|
| Opener | 0.4 | 0.3 | 0.3 |
| Carrying Conversation | 0.3 | 0.5 | 0.2 |
| LinkedIn Connection | 0.6 | 0.3 | 0.1 |
| Move On | 0.2 | 0.2 | 0.6 |
| Farewell | 0.1 | 0.5 | 0.4 |

These weights determine how points earned in each stage contribute to the three skill areas. For example, if a user earns 3 points in the Opener stage, they would receive:
- 1.2 points (3 × 0.4) toward Critical Thinking
- 0.9 points (3 × 0.3) toward Communication
- 0.9 points (3 × 0.3) toward Emotional Intelligence

### Badge Levels

One badge per lesson with three achievement levels based on overall performance:
- **Bronze** (Beginner): 1-5 points
- **Silver** (Intermediate): 6-10 points
- **Gold** (Advanced): 11-15 points

**Important**: To achieve a higher badge level, users must demonstrate minimum competency across all three skill areas. The specific thresholds for each skill area are:

| Badge Level | Minimum Critical Thinking | Minimum Communication | Minimum Emotional Intelligence |
|-------------|---------------------------|----------------------|-------------------------------|
| Bronze | 1 point | 1 point | 1 point |
| Silver | 3 points | 3 points | 3 points |
| Gold | 5 points | 5 points | 5 points |

A user with high scores in one area but scores below the threshold in another will be limited to the lowest badge level for which they meet all minimum requirements.

## Success Criteria by Skill Area

### Critical Thinking (Successful When You...)
- Identify and adapt to the other person's level of engagement
- Make logical, relevant connections between your work and theirs
- Anticipate possible responses and adjust your approach accordingly
- Recognize when a conversation isn't progressing and pivot strategically
- Ask insightful, open-ended questions that encourage deeper discussion
- Use the most effective tactic/language to reach your goals

### Communication (Successful When You...)
- Express your thoughts clearly and concisely without over-explaining
- Use approachable, engaging language that fits the context
- Make your ask (e.g., LinkedIn connection) feel natural, not transactional
- Ensure a smooth flow in conversation, avoiding abrupt topic shifts
- Balance talking and listening, keeping the conversation dynamic
- Ask questions of the other person to go deeper on what they said

### Emotional Intelligence (Successful When You...)
- Read verbal and nonverbal cues to gauge interest and engagement
- Mirror energy and tone to create a comfortable interaction
- Adjust your approach based on the other person's mood or behavior
- Show genuine curiosity and interest in their work, not just your goals
- Know when to exit a conversation gracefully without making it awkward
- Be polite
- Talk about topics which are non-controversial
- Be open-minded
- Talk about topics which most people are familiar with

## Technical Implementation

### Evaluation Engine

1. **Conversation Parser**
   - Breaks down conversation into stages
   - Identifies key phrases and conversational patterns
   - Maps user responses to scoring criteria

2. **Scoring Algorithm**
   - Evaluates each response based on predefined criteria
   - Calculates cumulative scores for each dimension using the numerical weights
   - Determines badge levels based on numerical thresholds
   - Ensures minimum competency across all three skill areas

3. **Feedback Generator**
   - Creates specific, actionable feedback for improvement
   - Highlights strengths and areas for development
   - Suggests alternative approaches for suboptimal responses

### Database Schema

```
Evaluation
  - id: UUID
  - user_id: Foreign Key
  - conversation_id: Foreign Key
  - timestamp: DateTime
  - critical_thinking_score: Float
  - communication_score: Float
  - emotional_intelligence_score: Float
  - total_score: Integer
  - badge_level: String
  - badges_earned: JSON

EvaluationDetail
  - id: UUID
  - evaluation_id: Foreign Key
  - stage: String (opener, carry, linkedin, moveon, farewell)
  - user_response: Text
  - points_awarded: Integer
  - feedback: Text
  - improvement_suggestions: Text
```

## Synthetic Testing System

Before full product integration, we'll develop a synthetic testing framework to validate the evaluation engine:

### Synthetic Conversation Generator

1. **Test Data Generation**
   - Create a library of persona responses for the AI conversation partner
   - Develop a set of user response templates at varying quality levels (optimal, good, needs improvement)
   - Generate complete synthetic conversations by combining these elements

2. **Test Harness**
   - Command-line tool to run synthetic conversations through the evaluation engine
   - Outputs detailed scoring reports and identifies potential issues
   - Allows for batch processing of multiple conversation scenarios

### Example Synthetic Conversation Structure
```json
{
  "conversation_id": "synthetic-test-001",
  "persona": "jake",
  "scenario": "networking_event",
  "exchanges": [
    {
      "stage": "opener",
      "ai_prompt": "Hi there! I'm Jake, I work in software development.",
      "user_response": "What brings you here today?",
      "expected_score": 3,
      "expected_feedback": "Excellent opener using open-ended question"
    },
    {
      "stage": "carry",
      "ai_prompt": "I'm here for the tech networking. I work on AI systems at TechCorp.",
      "user_response": "What got you started in AI?",
      "expected_score": 3,
      "expected_feedback": "Great follow-up question showing genuine interest"
    }
    // Additional exchanges for remaining stages
  ]
}
```

## Implementation Plan

### Phase 1: Core Evaluation Engine
- Develop the conversation parser
- Implement scoring algorithms based on framework criteria 
- Create basic feedback generation system

### Phase 2: Synthetic Testing System
- Build conversation generator with various quality levels
- Develop test harness to validate evaluation engine
- Create reporting tools to analyze results

### Phase 3: Database Integration
- Implement database schema for storing evaluations
- Connect evaluation engine to existing conversation system
- Create admin interface for reviewing evaluation results

### Phase 4: User Interface
- Design and implement post-conversation evaluation display
- Develop badge system and visualization
- Create personalized improvement recommendations

## Testing Criteria

The synthetic testing system should validate:

1. **Accuracy**: Scores match expected outcomes for pre-classified responses
2. **Consistency**: Similar responses receive similar scores across different contexts
3. **Comprehensiveness**: All aspects of the evaluation framework are properly assessed
4. **Feedback Quality**: Generated feedback is specific, actionable and helpful

## Extension Possibilities

- **Machine Learning Enhancement**: Train models to improve scoring accuracy based on expert-rated conversations
- **Comparative Analytics**: Show users how their performance compares to peers
- **Personalized Learning Paths**: Generate customized practice scenarios based on areas needing improvement
- **Real-time Evaluation**: Provide subtle hints during conversations for immediate improvement 