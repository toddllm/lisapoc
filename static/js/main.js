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
    document.getElementById('end-btn').style.display = 'none';  // Hide end button
    fetch('/feedback', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            const feedbackDiv = document.getElementById('feedback');
            feedbackDiv.style.display = 'block';
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
            `;
            document.getElementById('restart-btn').style.display = 'inline';
        });
}

document.getElementById('start-btn').addEventListener('click', () => {
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('end-btn').style.display = 'inline';  // Show end button
    document.getElementById('input-area').style.display = 'block';
    startTimer();
    fetch('/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            displayMessage('Alex', data.response);
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
            displayMessage('Alex', data.response);
            playAudio(data.audio);
            if (data.response.toLowerCase().includes('linkedin.com/in/alex') || timeLeft <= 0) {
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
