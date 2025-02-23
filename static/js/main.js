let recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US';
recognition.continuous = false;
recognition.interimResults = false;
let audio, animationInterval, timerInterval, timeLeft = 300; // 5 minutes in seconds

// Add push-to-talk functionality
const speakButton = document.getElementById('push-to-talk');
let isRecording = false;

// Replace push-to-talk with toggle functionality
speakButton.addEventListener('click', () => {
    if (!isRecording) {
        // Start recording
        if (audio && !audio.paused) {
            audio.pause();
            clearInterval(animationInterval);
            document.getElementById('mouth').setAttribute('d', 'M40 60 Q50 70 60 60');
        }
        
        try {
            recognition.start();
            speakButton.textContent = 'Stop Speaking';
            speakButton.classList.add('active');
            isRecording = true;
        } catch (err) {
            console.error('Recognition failed to start:', err);
        }
    } else {
        // Stop recording
        try {
            recognition.stop();
            speakButton.textContent = 'Start Speaking';
            speakButton.classList.remove('active');
            isRecording = false;
        } catch (err) {
            console.error('Recognition failed to stop:', err);
        }
    }
});

function startTimer() {
    timerInterval = setInterval(() => {
        timeLeft--;
        const minutes = Math.floor(timeLeft / 60);
        const seconds = timeLeft % 60;
        document.getElementById('timer').textContent = `Time Left: ${minutes}:${seconds < 10 ? '0' + seconds : seconds}`;
        if (timeLeft <= 0) {
            endConversation();
        }
    }, 1000);
}

function endConversation() {
    clearInterval(timerInterval);
    recognition.stop();
    document.getElementById('input-area').style.display = 'none';
    document.getElementById('end-btn').style.display = 'none';
    
    // Show loading indicator
    const feedbackDiv = document.getElementById('feedback');
    feedbackDiv.style.display = 'block';
    feedbackDiv.innerHTML = `
        <div class="loading-feedback">
            <h2>Generating Feedback...</h2>
            <div class="progress-bar">
                <div class="progress-bar-fill"></div>
            </div>
        </div>
    `;
    
    // Start progress bar animation
    const progressBar = document.querySelector('.progress-bar-fill');
    progressBar.style.width = '0%';
    setTimeout(() => progressBar.style.width = '100%', 100);
    
    fetch('/feedback', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            feedbackDiv.innerHTML = `
                <h2>Feedback</h2>
                <p>Goal: ${data.goal}</p>
                <p>Coins Earned: ${data.coins}</p>
                <h3>Critical Thinking</h3>
                <p>Strengths: ${data.critical_thinking.strengths}</p>
                <p>Improvement: ${data.critical_thinking.improvement}</p>
                <h3>Communication</h3>
                <p>Strengths: ${data.communication.strengths}</p>
                <p>Improvement: ${data.communication.improvement}</p>
                <h3>Emotional Intelligence</h3>
                <p>Strengths: ${data.emotional_intelligence.strengths}</p>
                <p>Improvement: ${data.emotional_intelligence.improvement}</p>
                <div class="end-buttons">
                    <button onclick="restartConversation()" class="restart-button">Restart Conversation</button>
                    <button onclick="returnToSelection()" class="selection-button">Choose New Partner</button>
                </div>
            `;
        });
}

function returnToSelection() {
    // Hide feedback and conversation
    document.getElementById('feedback').style.display = 'none';
    document.getElementById('conversation').innerHTML = '';
    
    // Show persona selector
    document.querySelector('.persona-selector').style.display = 'block';
    
    // Hide avatar container until new selection
    document.getElementById('avatar-container').style.display = 'none';
    
    // Reset timer display
    document.getElementById('timer').textContent = 'Time Left: 5:00';
    timeLeft = 300;
}

function restartConversation() {
    // Hide feedback
    document.getElementById('feedback').style.display = 'none';
    
    // Clear conversation
    document.getElementById('conversation').innerHTML = '';
    
    // Show avatar and start button
    document.getElementById('avatar-container').style.display = 'block';
    document.getElementById('start-btn').style.display = 'block';
    
    // Reset timer display
    document.getElementById('timer').textContent = 'Time Left: 5:00';
    timeLeft = 300;
}

document.getElementById('start-btn').addEventListener('click', () => {
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('end-btn').style.display = 'inline';
    document.getElementById('input-area').style.display = 'block';
    startTimer();
    fetch('/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            // Use the current persona name
            const currentPersona = document.querySelector('#avatar-container h2').textContent;
            displayMessage(currentPersona, data.response);
            playAudio(data.audio);
        });
});

document.getElementById('send-btn').addEventListener('click', () => {
    const input = document.getElementById('user-input').value;
    if (input) {
        displayMessage('You', input);
        sendInput(input);
        document.getElementById('user-input').value = '';
    }
});

recognition.onaudiostart = () => {
    if (audio && !audio.paused) {
        audio.pause();
        clearInterval(animationInterval);
        document.getElementById('mouth').setAttribute('d', 'M40 60 Q50 70 60 60');
    }
};

recognition.onresult = (event) => {
    const results = event.results;
    const transcript = results[results.length - 1][0].transcript;
    displayMessage('You', transcript);
    sendInput(transcript);
    
    // Stop recording after getting a result
    recognition.stop();
    speakButton.textContent = 'Start Speaking';
    speakButton.classList.remove('active');
    isRecording = false;
};

recognition.onend = () => {
    // Always reset button state when recognition ends
    speakButton.textContent = 'Start Speaking';
    speakButton.classList.remove('active');
    isRecording = false;
};

recognition.onerror = (event) => {
    console.error('Speech recognition error', event.error);
};

document.getElementById('restart-btn').addEventListener('click', () => {
    location.reload();
});

function sendInput(input) {
    fetch('/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: input })
    })
        .then(response => response.json())
        .then(data => {
            // Use the current persona name instead of hardcoded "Alex"
            const currentPersona = document.querySelector('#avatar-container h2').textContent;
            displayMessage(currentPersona, data.response);
            playAudio(data.audio);
            if (data.response.toLowerCase().includes('linkedin') || timeLeft <= 0) {
                endConversation();
            }
        });
}

function displayMessage(sender, message) {
    const conversation = document.getElementById('conversation');
    conversation.innerHTML += `<p><strong>${sender}:</strong> ${message}</p>`;
    conversation.scrollTop = conversation.scrollHeight;
}

function playAudio(audioUrl) {
    audio = new Audio(audioUrl);
    audio.onplay = () => {
        animationInterval = setInterval(animateMouth, 400);
    };
    audio.onended = () => {
        clearInterval(animationInterval);
        document.getElementById('mouth').setAttribute('d', 'M40 60 Q50 70 60 60');
    };
    audio.play();
}

function animateMouth() {
    const mouth = document.getElementById('mouth');
    mouth.setAttribute('d', 'M40 60 Q50 65 60 60');
    setTimeout(() => mouth.setAttribute('d', 'M40 60 Q50 70 60 60'), 200);
}

// Add end button handler
document.getElementById('end-btn').addEventListener('click', () => {
    // Stop recording if it's active
    if (isRecording) {
        recognition.stop();
        speakButton.textContent = 'Start Speaking';
        speakButton.classList.remove('active');
        isRecording = false;
    }
    
    // Stop any playing audio
    if (audio && !audio.paused) {
        audio.pause();
        clearInterval(animationInterval);
        document.getElementById('mouth').setAttribute('d', 'M40 60 Q50 70 60 60');
    }
    
    document.getElementById('end-btn').style.display = 'none';
    endConversation();
});

// Add persona selection functionality
function selectPersona(personaId) {
    fetch('/select_persona', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ persona_id: personaId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Selected persona:', data.persona); // Debug log
            
            // Show avatar container first
            const avatarContainer = document.getElementById('avatar-container');
            avatarContainer.style.display = 'block';
            
            // Update avatar appearance
            updateAvatar(data.persona.avatar);
            
            // Update name display
            avatarContainer.querySelector('h2').textContent = data.persona.name;
            
            // Update page title to match selected persona
            document.querySelector('.role-subtitle').textContent = data.persona.role;
            
            // Hide persona selector and show start button
            document.querySelector('.persona-selector').style.display = 'none';
            document.getElementById('start-btn').style.display = 'block';
            
            // Reset conversation history
            document.getElementById('conversation').innerHTML = '';
        } else {
            console.error('Failed to select persona:', data.message);
        }
    })
    .catch(error => {
        console.error('Error selecting persona:', error);
    });
}

function updateAvatar(avatarConfig) {
    const avatar = document.getElementById('avatar');
    
    // Clear existing avatar content
    while (avatar.firstChild) {
        avatar.removeChild(avatar.firstChild);
    }
    
    // Create base circle
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute('cx', '50');
    circle.setAttribute('cy', '50');
    circle.setAttribute('r', '40');
    circle.setAttribute('fill', avatarConfig.color);
    avatar.appendChild(circle);
    
    // Add eyes
    const leftEye = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    leftEye.setAttribute('cx', '35');
    leftEye.setAttribute('cy', '40');
    leftEye.setAttribute('r', '5');
    leftEye.setAttribute('fill', 'black');
    avatar.appendChild(leftEye);
    
    const rightEye = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    rightEye.setAttribute('cx', '65');
    rightEye.setAttribute('cy', '40');
    rightEye.setAttribute('r', '5');
    rightEye.setAttribute('fill', 'black');
    avatar.appendChild(rightEye);
    
    // Add hair if needed
    if (avatarConfig.hair === 'long') {
        const hair = document.createElementNS("http://www.w3.org/2000/svg", "path");
        hair.setAttribute('d', 'M30 30 C30 50 70 50 70 30 L70 60 C70 80 30 80 30 60 Z');
        hair.setAttribute('fill', '#4a4a4a');
        hair.setAttribute('data-type', 'hair');
        avatar.insertBefore(hair, circle);
    }
    
    // Add mouth
    const mouth = document.createElementNS("http://www.w3.org/2000/svg", "path");
    mouth.setAttribute('id', 'mouth');
    mouth.setAttribute('stroke', 'black');
    mouth.setAttribute('fill', 'none');
    if (avatarConfig.expression === 'professional') {
        mouth.setAttribute('d', 'M40 60 Q50 65 60 60');
    } else {
        mouth.setAttribute('d', 'M40 60 Q50 70 60 60');
    }
    avatar.appendChild(mouth);
}

// Show persona selector initially, hide start button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('start-btn').style.display = 'none';
});
