// Get references to all the HTML elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusEl = document.getElementById('status');
const homeScreen = document.getElementById('home-screen');
const appContainer = document.getElementById('app-container');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

// Emotion bar elements
const happyBar = document.getElementById('happy-bar');
const sadBar = document.getElementById('sad-bar');
const angryBar = document.getElementById('angry-bar');
const neutralBar = document.getElementById('neutral-bar');
const surprisedBar = document.getElementById('surprised-bar');

// Global variables to hold the detection interval and video stream
let detectionInterval;
let videoStream;

// --- UI UPDATE FUNCTIONS ---

function updateEmotionUI(expressions) {
    if (!expressions) return;
    happyBar.style.width = `${expressions.happy * 100}%`;
    sadBar.style.width = `${expressions.sad * 100}%`;
    angryBar.style.width = `${expressions.angry * 100}%`;
    neutralBar.style.width = `${expressions.neutral * 100}%`;
    surprisedBar.style.width = `${expressions.surprised * 100}%`;

    let highestEmotion = Object.keys(expressions).reduce((a, b) => expressions[a] > expressions[b] ? a : b);
    let confidence = Math.round(expressions[highestEmotion] * 100);
    highestEmotion = highestEmotion.charAt(0).toUpperCase() + highestEmotion.slice(1);
    statusEl.innerText = `Emotion: ${highestEmotion} (${confidence}%)`;
}

function resetEmotionUI() {
    statusEl.innerText = 'Click Start to begin';
    const emotions = { happy: 0, sad: 0, angry: 0, neutral: 0, surprised: 0 };
    updateEmotionUI(emotions);
}

// --- CORE APP FUNCTIONS ---

// Function to start the webcam and detection
async function startApp() {
    // 1. Show the app and hide the home screen
    homeScreen.classList.add('hidden');
    appContainer.classList.remove('hidden');
    statusEl.innerText = 'Starting Webcam...';

    // 2. Start the webcam stream
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: {} });
        video.srcObject = videoStream;
    } catch (err) {
        console.error('Error accessing webcam:', err);
        statusEl.innerText = 'Error: Could not access webcam.';
        stopApp(); // Go back to home screen on error
    }
}

// Function to stop the webcam and detection
function stopApp() {
    // 1. Stop the detection loop
    clearInterval(detectionInterval);

    // 2. Stop the webcam stream
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }

    // 3. Clear the canvas and reset UI
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    resetEmotionUI();

    // 4. Show the home screen and hide the app
    appContainer.classList.add('hidden');
    homeScreen.classList.remove('hidden');
}

// --- EVENT LISTENERS ---

// When the video starts playing, set up the detection interval
video.addEventListener('play', () => {
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);
    statusEl.innerText = 'Detecting...';

    detectionInterval = setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceExpressions();

        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        if (detections.length > 0) {
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            updateEmotionUI(resizedDetections[0].expressions);
        } else {
            statusEl.innerText = 'No face detected';
        }
    }, 100);
});

// Add click listeners to buttons
startButton.addEventListener('click', startApp);
stopButton.addEventListener('click', stopApp);

// --- INITIALIZATION ---

// Load models when the script first runs
async function loadModels() {
    statusEl.innerText = 'Loading Models...';
    try {
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
            faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
            faceapi.nets.faceExpressionNet.loadFromUri('/models')
        ]);
        resetEmotionUI();
    } catch (error) {
        console.error("Failed to load models", error);
        statusEl.innerText = 'Error: Failed to load models.';
    }
}

loadModels();