#!/usr/bin/env python3
"""
Generate a single conversation using the OpenAI API.
This is a simplified version of the test to help debug issues.
"""

import os
import sys
import json
import time
import requests

def generate_conversation():
    """Generate a single conversation using the OpenAI API."""
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")
    
    # Set up the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Create a simple system prompt
    system_prompt = """
    You are simulating a networking conversation at a professional event.

    You will play two roles:
    1. Jake, a Software Engineer at TechCorp who specializes in AI systems with 5 years of experience
    2. A person attending the networking event who is a novice at networking

    Generate a realistic conversation between these two people that includes all five stages:
    1. Opener: Initial greeting and conversation starter
    2. Carrying Conversation: Maintaining dialogue flow and asking about the other person
    3. LinkedIn Connection: Asking to connect professionally
    4. Move On: Gracefully transitioning away from the conversation
    5. Farewell: Closing the conversation politely

    Format your response as a JSON object with a "conversation" field containing an array of exchanges, where each exchange has:
    - "stage": The conversation stage (opener, carry, linkedin, moveon, farewell)
    - "ai_prompt": What Jake says
    - "user_response": What the networking person says
    """
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a realistic networking conversation between Jake and a novice networker."}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }
    
    # Make the API request
    print("\nSending request to generate a conversation...")
    print("This may take 15-30 seconds...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                conversation_data = json.loads(content)
                conversation = conversation_data.get("conversation", [])
                
                # Save the conversation to a file
                output_file = "debug_conversation.json"
                with open(output_file, "w") as f:
                    json.dump(conversation_data, f, indent=2)
                
                print(f"\nConversation generated and saved to {output_file}")
                print(f"Generated {len(conversation)} exchanges")
                
                # Print a sample of the conversation
                if conversation:
                    print("\nSample exchange:")
                    exchange = conversation[0]
                    print(f"Stage: {exchange.get('stage', 'unknown')}")
                    print(f"AI: {exchange.get('ai_prompt', '')}")
                    print(f"User: {exchange.get('user_response', '')}")
                
                return True
            except json.JSONDecodeError as e:
                print(f"Error parsing response as JSON: {e}")
                print(f"Raw response: {content}")
                return False
        else:
            print(f"Error: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("Error: Request timed out after 60 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Generate Single Conversation")
    print("===========================")
    
    success = generate_conversation()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 