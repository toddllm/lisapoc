#!/usr/bin/env python3
"""
Debug script to test OpenAI API connection.
"""

import os
import sys
import json
import time
import requests

def test_openai_api():
    """Test the OpenAI API connection with a simple request."""
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
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello world!"}
        ],
        "temperature": 0.7
    }
    
    # Make the API request
    print("\nSending test request to OpenAI API...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            message = response_data["choices"][0]["message"]["content"]
            print(f"\nResponse: {message}")
            print("\nAPI connection successful!")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("Error: Request timed out after 30 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("OpenAI API Connection Test")
    print("=========================")
    
    success = test_openai_api()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 