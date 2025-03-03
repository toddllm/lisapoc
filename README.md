# Networking Conversation Generator and Evaluator

## Overview
This script generates and evaluates networking conversations using OpenAI's GPT models. It creates conversations for different skill levels and provides detailed evaluations based on a comprehensive framework.

## Features
- Generates realistic networking conversations for 9 skill levels
- Evaluates conversations across multiple dimensions:
  - Critical Thinking
  - Communication
  - Emotional Intelligence
- Provides stage-by-stage analysis of conversation performance
- Assigns badge levels based on overall performance
- Offers specific feedback and improvement suggestions

## Usage
```bash
python generate_simple.py
```

## Output
The script generates two files:
- `simple_output/conversations.txt`: Contains the generated conversations and evaluations
- `simple_output/debug.txt`: Contains debugging information

## Evaluation Framework
The evaluation is based on a comprehensive framework that assesses:
1. Conversation stages (Opener, Carrying Conversation, LinkedIn Connection, Move On, Farewell)
2. Skill dimensions (Critical Thinking, Communication, Emotional Intelligence)
3. Overall performance with badge levels (Bronze, Silver, Gold)

For more details, see `evaluation_design.md`.

## Testing
Run the tests with:
```bash
python -m unittest test_generate_simple.py
```

## Sample Conversations
Sample conversations for testing and validation are available in the `sample_conversations` directory.

## Development
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Implement your changes
5. Run the tests
6. Submit a pull request
