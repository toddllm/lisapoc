from flask import Flask, render_template, request, jsonify, session
import os
import time
from openai import OpenAI
from TTS.api import TTS
import torch
import json
from datetime import datetime
import uuid

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Initialize TTS model with faster, more natural-sounding settings
tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
tts_model.to("cuda" if torch.cuda.is_available() else "cpu")

def load_personas():
    try:
        with open('personas.json', 'r') as f:
            personas = json.load(f)
            # Force update Jake's avatar every time
            if 'jake' in personas:
                personas['jake']['avatar'] = {
                    "type": "image",
                    "url": "/static/images/neutral-male-processed.png"
                }
            return personas
    except FileNotFoundError:
        print("Warning: personas.json not found, using default personas")
        return {}

def save_personas(personas):
    with open('personas.json', 'w') as f:
        json.dump(personas, f, indent=4)

# Define system prompts first
SYSTEM_PROMPTS = {
    "jake": """You are Jake, a software engineer at a networking event. Keep responses natural and focused on making a genuine connection.

Key behaviors:
- Give concise, natural responses (1-2 sentences)
- Show genuine interest in the other person's work
- Share relevant experiences briefly
- Ask thoughtful follow-up questions
- Avoid repeating yourself
- When appropriate, suggest connecting on LinkedIn

Example flow:
1. Start with a friendly greeting and mention your role
2. Ask about their work and listen actively
3. Share relevant experiences or insights
4. Build rapport through common interests
5. Suggest LinkedIn connection when rapport is established""",

    "sarah": """You are Sarah, a product manager at a networking event. Keep responses natural and focused on making a genuine connection.

Key behaviors:
- Give concise, natural responses (1-2 sentences)
- Show genuine interest in others' work and challenges
- Share brief insights about product development
- Ask thoughtful follow-up questions
- Avoid repeating yourself
- When appropriate, suggest connecting on LinkedIn

Example flow:
1. Start with a professional greeting and mention your role
2. Ask about their work and specific challenges
3. Share relevant product management insights
4. Build rapport through shared industry experiences
5. Suggest LinkedIn connection when rapport is established"""
}

# Load personas
PERSONAS = load_personas()

# Update avatars in loaded personas
if 'jake' in PERSONAS:
    PERSONAS['jake']['avatar'] = {
        "type": "image",
        "url": "/static/images/neutral-male-processed.png"
    }

if 'sarah' in PERSONAS:
    PERSONAS['sarah']['avatar'] = {
        "type": "image",
        "url": "/static/images/neutral-female-processed.png"
    }

# If no personas loaded, create default ones
if not PERSONAS:
    PERSONAS = {
        "jake": {
            "name": "Jake",
            "role": "Software Engineer",
            "voice": "p273",
            "speed": 1.3,
            "avatar": {
                "type": "image",
                "url": "/static/images/neutral-male-processed.png"
            },
            "description": "A friendly software engineer specializing in AI and machine learning. Jake is enthusiastic about tech and loves connecting with fellow developers.",
            "system_prompt": SYSTEM_PROMPTS["jake"]
        },
        "sarah": {
            "name": "Sarah",
            "role": "Product Manager",
            "voice": "p226",
            "speed": 1.4,
            "avatar": {
                "type": "image",
                "url": "/static/images/neutral-female-processed.png"
            },
            "description": "A strategic product manager with experience in tech startups. Sarah is great at understanding user needs and building relationships.",
            "system_prompt": SYSTEM_PROMPTS["sarah"]
        }
    }
    save_personas(PERSONAS)

# Update system prompts in personas
for persona_id in PERSONAS:
    PERSONAS[persona_id]['system_prompt'] = SYSTEM_PROMPTS.get(persona_id, '')

FEEDBACK_PROMPT = """Analyze this networking conversation and provide specific, constructive feedback.
Focus on:
1. Goal achievement (making a genuine connection and exchanging LinkedIn profiles)
2. Communication effectiveness
3. Active listening and engagement
4. Professional relationship building

Compute coins (1-10) based on these criteria:
• +2 coins: Clear introduction and role statement
• +2 coins: Active listening with relevant follow-up questions
• +2 coins: Sharing relevant professional experiences
• +2 coins: Building genuine rapport/common ground
• +2 coins: Successfully exchanging LinkedIn contacts

Format the response as JSON with these keys:
{
    "goal_status": "Completed" or "Incomplete",
    "coins": number from 1-10 based on criteria above,
    "strengths": [
        "• Clear introduction and role statement",
        "• Active listening with follow-up questions",
        "• Genuine interest in shared topics"
    ],
    "improvements": [
        "• More specific questions about work experience",
        "• Earlier focus on common professional interests",
        "• Direct LinkedIn connection request"
    ],
    "key_moments": [
        "• Strong opening with clear introduction",
        "• Missed opportunity for deeper technical discussion"
    ]
}

Style guide:
- Start each point with a bullet (•)
- Use short, actionable phrases
- Focus on specific behaviors and moments
- Avoid complete sentences
- Keep each point under 8 words"""

def get_openai_response(history):
    print("Current conversation history:")
    for msg in history:
        print(f"{msg['role']}: {msg['content'][:50]}...")
    
    # Ensure the first message is always the system prompt
    if not history or history[0]['role'] != 'system':
        current_persona = session.get('current_persona', 'jake')
        history.insert(0, {'role': 'system', 'content': PERSONAS[current_persona]['system_prompt']})
    
    # Keep conversation context manageable
    if len(history) > 11:  # system prompt + 10 messages
        history = [history[0]] + history[-10:]  # Keep system prompt and last 10 messages
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        temperature=0.7,
        max_tokens=100,
        presence_penalty=0.6,
        frequency_penalty=0.8  # Increased to further reduce repetition
    )
    return response.choices[0].message.content

def generate_audio(text):
    current_persona = session.get('current_persona', 'jake')
    persona = PERSONAS[current_persona]
    
    print(f"Using voice: {persona['voice']} for {persona['name']}")  # Debug print
    
    audio_file = f"static/audio/response_{int(time.time())}.wav"
    tts_model.tts_to_file(
        text=text,
        file_path=audio_file,
        speaker=persona['voice'],
        speed=persona['speed']
    )
    return audio_file

def generate_feedback(conversation_history):
    # Prepare conversation summary for feedback
    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in conversation_history
        if msg['role'] != 'system'
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": FEEDBACK_PROMPT},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.7,
            max_tokens=500  # Increased token limit for fuller feedback
        )
        
        # Add error handling for JSON parsing
        try:
            feedback = json.loads(response.choices[0].message.content)
            print("Generated feedback:", feedback)  # Debug print
            return feedback
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("Raw response:", response.choices[0].message.content)
            return {
                "goal_status": "Incomplete",
                "coins": 5,
                "strengths": ["Participated in conversation"],
                "improvements": ["Technical error in feedback generation"],
                "key_moments": []
            }
            
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return {
            "goal_status": "Incomplete",
            "coins": 5,
            "strengths": ["Conversation attempted"],
            "improvements": ["Error in feedback generation"],
            "key_moments": []
        }

@app.route('/')
def index():
    # Start with Jake as default persona
    default_persona = PERSONAS['jake']
    session['current_persona'] = 'jake'
    session['history'] = [{'role': 'system', 'content': PERSONAS['jake']['system_prompt']}]
    return render_template('index.html', personas=PERSONAS, persona=default_persona)

@app.route('/select_persona', methods=['POST'])
def select_persona():
    persona_id = request.json.get('persona_id')
    if persona_id in PERSONAS:
        session['current_persona'] = persona_id
        # Initialize history with system prompt
        session['history'] = [{'role': 'system', 'content': PERSONAS[persona_id]['system_prompt']}]
        session.modified = True  # Ensure session is saved
        return jsonify({
            'status': 'success',
            'persona': PERSONAS[persona_id]
        })
    return jsonify({'status': 'error', 'message': 'Invalid persona'}), 400

@app.route('/start', methods=['POST'])
def start_conversation():
    current_persona = session.get('current_persona', 'jake')
    
    # Reset history to just the system prompt
    session['history'] = [{'role': 'system', 'content': PERSONAS[current_persona]['system_prompt']}]
    session.modified = True
    
    # Don't generate initial greeting, just return success
    return jsonify({
        'status': 'success',
        'persona': PERSONAS[current_persona]['name']
    })

@app.route('/respond', methods=['POST'])
def respond():
    current_persona = session.get('current_persona', 'jake')
    user_input = request.json.get('input', '')
    
    if 'history' not in session:
        # Initialize with system prompt if no history exists
        session['history'] = [{'role': 'system', 'content': PERSONAS[current_persona]['system_prompt']}]
    
    # Add user message
    session['history'].append({'role': 'user', 'content': user_input})
    
    # Get AI response
    response_text = get_openai_response(session['history'])
    
    # Add AI response to history
    session['history'].append({'role': 'assistant', 'content': response_text})
    session.modified = True
    
    # Generate audio
    audio_file = generate_audio(response_text)
    
    return jsonify({
        'response': response_text,
        'audio': audio_file,
        'persona': PERSONAS[current_persona]['name']
    })

@app.route('/end-conversation', methods=['POST'])
def end_conversation():
    # Clear the conversation history but keep the current persona
    current_persona = session.get('current_persona', 'jake')
    session['history'] = [{'role': 'system', 'content': PERSONAS[current_persona]['system_prompt']}]
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/conversations')
def view_conversations():
    conversations = get_conversations()
    return render_template('conversations.html', conversations=conversations)

@app.route('/conversation/<conversation_id>')
def view_conversation(conversation_id):
    conversations = get_conversations()
    conversation = next((c for c in conversations if c['id'] == conversation_id), None)
    if conversation:
        return render_template('conversation.html', conversation=conversation)
    return "Conversation not found", 404

@app.route('/feedback', methods=['POST'])
def feedback():
    if 'history' not in session:
        print("No history found in session")
        return jsonify({
            "goal_status": "Incomplete",
            "coins": 0,
            "critical_thinking": {
                "strengths": "No conversation recorded",
                "improvement": "Start a conversation first"
            },
            "communication": {
                "strengths": "N/A",
                "improvement": "N/A"
            },
            "emotional_intelligence": {
                "strengths": "N/A",
                "improvement": "N/A"
            }
        })

    # Save conversation before generating feedback
    current_persona = session.get('current_persona', 'jake')
    conversation_id = save_conversation(PERSONAS[current_persona]['name'], session['history'])
    
    print("Generating feedback for conversation history:")
    for msg in session['history']:
        print(f"{msg['role']}: {msg['content'][:50]}...")

    feedback_response = generate_feedback(session['history'])
    
    # Convert the new feedback format to match the frontend expectations
    formatted_feedback = {
        "goal": feedback_response.get("goal_status", "Incomplete"),
        "coins": feedback_response.get("coins", 5),
        "critical_thinking": {
            "strengths": feedback_response.get("strengths", [""])[0] if feedback_response.get("strengths") else "",
            "improvement": feedback_response.get("improvements", [""])[0] if feedback_response.get("improvements") else ""
        },
        "communication": {
            "strengths": feedback_response.get("strengths", ["", ""])[1] if len(feedback_response.get("strengths", [])) > 1 else "",
            "improvement": feedback_response.get("improvements", ["", ""])[1] if len(feedback_response.get("improvements", [])) > 1 else ""
        },
        "emotional_intelligence": {
            "strengths": feedback_response.get("strengths", ["", "", ""])[2] if len(feedback_response.get("strengths", [])) > 2 else "",
            "improvement": feedback_response.get("improvements", ["", "", ""])[2] if len(feedback_response.get("improvements", [])) > 2 else ""
        }
    }
    
    print("Formatted feedback:", formatted_feedback)  # Debug print
    return jsonify(formatted_feedback)

# Add new admin routes
@app.route('/admin')
def admin():
    # Get list of available voices from TTS model
    available_voices = tts_model.speakers
    return render_template('admin.html', 
                         personas=PERSONAS,
                         available_voices=available_voices)

@app.route('/admin/test-voice', methods=['POST'])
def test_voice():
    voice = request.json.get('voice')
    speed = float(request.json.get('speed', 1.0))  # Ensure it's a float
    text = request.json.get('text', "This is a test of how this voice sounds.")
    
    print(f"Testing voice: {voice} at speed: {speed}")  # Debug print
    
    audio_file = f"static/audio/test_{int(time.time())}.wav"
    try:
        tts_model.tts_to_file(
            text=text,
            file_path=audio_file,
            speaker=voice,
            speed=speed,  # Make sure speed is being used
            # Add optional parameters that might help with speed
            length_scale=1.0/speed  # Adjust length scale inversely with speed
        )
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
    return jsonify({
        'status': 'success',
        'audio': audio_file
    })

@app.route('/admin/save-persona', methods=['POST'])
def save_persona():
    persona_id = request.json.get('persona_id')
    voice = request.json.get('voice')
    speed = float(request.json.get('speed'))
    
    if persona_id in PERSONAS:
        PERSONAS[persona_id]['voice'] = voice
        PERSONAS[persona_id]['speed'] = speed
        save_personas(PERSONAS)  # Save changes to file
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Invalid persona'}), 400

# Add conversation storage functions
def save_conversation(persona_name, history):
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    conversation = {
        'id': conversation_id,
        'persona': persona_name,
        'timestamp': timestamp,
        'history': history
    }
    
    try:
        with open('conversations.json', 'r') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        conversations = []
    
    conversations.append(conversation)
    
    with open('conversations.json', 'w') as f:
        json.dump(conversations, f, indent=4)
    
    return conversation_id

def get_conversations():
    try:
        with open('conversations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@app.template_filter('datetime')
def format_datetime(value):
    dt = datetime.fromisoformat(value)
    return dt.strftime('%B %d, %Y at %I:%M %p')  # Example: "March 15, 2024 at 02:30 PM"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)