let API_BASE = 'http://127.0.0.1:5000';

// --- CONNECTION CHECK ---
async function checkBackend() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('Backend connected:', data);
    } catch (e) {
        console.warn('Initial connection to 127.0.0.1 failed, trying localhost...');
        try {
            const fallback = await fetch('http://localhost:5000/health');
            const data = await fallback.json();
            API_BASE = 'http://localhost:5000';
            console.log('Switched to localhost');
        } catch (err) {
            console.error('Backend unreachable');
        }
    }
}
checkBackend();

// --- PREDICTION LOGIC ---
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const imageUpload = document.getElementById('imageUpload');
const resultCard = document.getElementById('resultCard');
const predictedDigit = document.getElementById('predictedDigit');
const predictionConfidence = document.getElementById('predictionConfidence');

// Set initial canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let isDrawing = false;
let lastX = 0;
let lastY = 0;

ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'black';

function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getPosition(e);
    [lastX, lastY] = [x, y];
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getPosition(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

function stopDrawing() { isDrawing = false; }

function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    return { x: clientX - rect.left, y: clientY - rect.top };
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultCard.style.display = 'none';
});

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                const x = (canvas.width / 2) - (img.width / 2) * scale;
                const y = (canvas.height / 2) - (img.height / 2) * scale;
                ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});

predictBtn.addEventListener('click', async () => {
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled = true;
    try {
        const imageData = canvas.toDataURL('image/png');
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            predictedDigit.textContent = data.prediction;
            predictionConfidence.textContent = (data.confidence * 100).toFixed(2) + '%';
            resultCard.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to backend. Is app.py running?');
    } finally {
        predictBtn.textContent = 'Predict';
        predictBtn.disabled = false;
    }
});


// --- GENERATOR LOGIC ---
const generateBtn = document.getElementById('generateBtn');
const generatedImg = document.getElementById('generatedImg');
const placeholderText = document.getElementById('placeholderText');
const digitSelect = document.getElementById('digitSelect');

generateBtn.addEventListener('click', async () => {
    generateBtn.textContent = 'Generating...';
    generateBtn.disabled = true;
    
    try {
        const selectedDigit = digitSelect.value;
        const url = selectedDigit !== "" 
            ? `${API_BASE}/generate?label=${selectedDigit}`
            : `${API_BASE}/generate`;

        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            placeholderText.style.display = 'none';
            generatedImg.src = data.image;
            generatedImg.style.display = 'block';
            
            // Add a little pop animation
            generatedImg.style.transform = 'scale(0.9)';
            setTimeout(() => {
                generatedImg.style.transition = 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                generatedImg.style.transform = 'scale(1)';
            }, 50);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to the backend server. Make sure app.py is running.');
    } finally {
        generateBtn.textContent = 'Generate New Digit';
        generateBtn.disabled = false;
    }
});
