function testVoice(personaId) {
    const voiceSelect = document.querySelector(`.voice-select[data-persona="${personaId}"]`);
    const speedSlider = document.querySelector(`.speed-slider[data-persona="${personaId}"]`);
    const testText = voiceSelect.closest('.persona-config-card').querySelector('.test-text').value;
    
    fetch('/admin/test-voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            voice: voiceSelect.value,
            speed: speedSlider.value,
            text: testText
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            const audio = new Audio(data.audio);
            audio.play();
        }
    });
}

function savePersonaSettings(personaId) {
    const voiceSelect = document.querySelector(`.voice-select[data-persona="${personaId}"]`);
    const speedSlider = document.querySelector(`.speed-slider[data-persona="${personaId}"]`);
    
    fetch('/admin/save-persona', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            persona_id: personaId,
            voice: voiceSelect.value,
            speed: speedSlider.value
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Settings saved successfully!');
        }
    });
}

// Update speed value display when slider moves
document.querySelectorAll('.speed-slider').forEach(slider => {
    const valueDisplay = slider.nextElementSibling;
    slider.addEventListener('input', () => {
        valueDisplay.textContent = slider.value;
    });
}); 