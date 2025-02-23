from flask import Flask, render_template, request, jsonify, session
from gtts import gTTS
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
app.secret_key = 'supersecretkey'

SYSTEM_PROMPT = """You are Jake, a friendly software engineer at a networking event. Your goal is to make a genuine connection and potentially exchange LinkedIn profiles.

Key traits:
- Keep responses casual and natural, like real conversation
- Use simple language, avoid corporate speak
- Keep responses short (1-2 sentences usually)
- Show genuine interest in the other person's work
- Be friendly but professional
- If they mention AI or tech, get genuinely excited and ask specific questions
- Aim to find common ground before asking to connect on LinkedIn

Example responses:
"Yeah, I've been working on some ML projects too! What kind of models are you using?"
"That's really interesting! I'm actually working on something similar at my company."
"Hey, sounds like we're in similar fields. Want to connect on LinkedIn to stay in touch?"

Remember: You're at a networking event having a natural conversation. Be authentic and interested, not salesy."""

def get_openai_response(history):
    # Debug print
    print("Current conversation history:")
    for msg in history:
        print(f"{msg['role']}: {msg['content'][:50]}...")
    
    # Make sure system prompt is first
    if not history or history[0]['role'] != 'system':
        history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    
    # Ensure history doesn't get too long (keep last 10 messages)
    if len(history) > 11:  # 1 system message + 10 conversation messages
        history = [history[0]] + history[-10:]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        temperature=0.9,
        max_tokens=100
    )
    return response.choices[0].message.content

@app.route('/')
def index():
    # Initialize conversation with system prompt
    session['history'] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_conversation():
    # Start with a natural greeting
    response_text = get_openai_response(session['history'])
    session['history'].append({'role': 'assistant', 'content': response_text})
    
    tts = gTTS(response_text)
    audio_file = f"static/audio/response_{int(time.time())}.mp3"
    tts.save(audio_file)
    
    return jsonify({
        'response': response_text, 
        'audio': audio_file
    })

@app.route('/respond', methods=['POST'])
def respond():
    user_input = request.json.get('input', '')
    
    # Initialize history if it doesn't exist
    if 'history' not in session:
        session['history'] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    
    # Get current history
    current_history = session['history']
    
    # Add user's message
    current_history.append({'role': 'user', 'content': user_input})
    
    # Get AI response
    response_text = get_openai_response(current_history)
    
    # Add AI's response
    current_history.append({'role': 'assistant', 'content': response_text})
    
    # Save back to session
    session['history'] = current_history
    session.modified = True  # Ensure Flask knows the session was modified
    
    tts = gTTS(response_text)
    audio_file = f"static/audio/response_{int(time.time())}.mp3"
    tts.save(audio_file)
    
    return jsonify({
        'response': response_text, 
        'audio': audio_file
    })

@app.route('/stop', methods=['POST'])
def stop_conversation():
    session['history'] = []
    return jsonify({'status': 'success'})

@app.route('/feedback', methods=['POST'])
def feedback():
    goal_achieved = any("linkedin" in msg['content'].lower() for msg in session['history'] if msg['role'] == 'assistant')
    coins = 10 if goal_achieved else 5
    feedback = {
        "goal": "Completed" if goal_achieved else "Incomplete",
        "coins": coins,
        "critical_thinking": {
            "strengths": "Good conversation flow",
            "improvement": "Try connecting over shared interests"
        },
        "communication": {
            "strengths": "Natural dialogue",
            "improvement": "Ask more follow-up questions"
        },
        "emotional_intelligence": {
            "strengths": "Friendly attitude",
            "improvement": "Show more curiosity about their work"
        }
    }
    return jsonify(feedback)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
