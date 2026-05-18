// Получаем элементы интерфейса
const recordBtn = document.getElementById("record-btn");
const micIcon = document.getElementById("mic-icon");
const hintText = document.getElementById("hint-text");
const statusText = document.getElementById("status-text");
const statusDot = document.querySelector(".status-dot");
const chatContainer = document.getElementById("chat-container");
const audioPlayer = document.getElementById("audio-player");
const canvas = document.getElementById("orb-waves");
const ctx = canvas.getContext("2d");
const chatInputForm = document.getElementById("chat-input-form");
const chatTextInput = document.getElementById("chat-text-input");

// Параметры боковой панели
const modeSelect = document.getElementById("mode-select");
const brainSelect = document.getElementById("brain-select");
const sttSelect = document.getElementById("stt-select");
const apiUrlInput = document.getElementById("api-url-input");
const voiceSelect = document.getElementById("voice-select");
const speedRange = document.getElementById("speed-range");
const speedVal = document.getElementById("speed-val");
const stepsRange = document.getElementById("steps-range");
const stepsVal = document.getElementById("steps-val");

// Обновление значений в UI
speedRange.addEventListener("input", (e) => speedVal.textContent = e.target.value);
stepsRange.addEventListener("input", (e) => stepsVal.textContent = e.target.value);

// Автозаполнение URL API при выборе ИИ
brainSelect.addEventListener("change", (e) => {
    const val = e.target.value;
    const group = document.getElementById("api-url-group");
    if (val === "vllm") {
        group.style.display = "block";
        apiUrlInput.value = "http://localhost:8000/v1";
    } else if (val === "lm-studio") {
        group.style.display = "block";
        apiUrlInput.value = "http://localhost:1234/v1";
    } else if (val === "ollama") {
        group.style.display = "block";
        apiUrlInput.value = "http://localhost:11434";
    } else {
        group.style.display = "none";
    }
});

// Инициализация Canvas
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// Состояния интерфейса
let isRecording = false;
let isSpeaking = false;
let recognition = null;
let animationFrameId = null;

// Настройка Web Audio API для визуализации голоса ИИ
let audioCtx = null;
let analyser = null;
let sourceNode = null;
let frequencyData = new Uint8Array(0);

// WebSocket и Аудио-очередь для реального времени
let ws = null;
let currentAssistantMessageElement = null;
let audioQueue = [];
let isPlayingAudio = false;
let currentSourceNode = null;
let serverDone = false;

// Глобальные переменные для режима Свободные руки (Real-Time)
let streamCtx = null;
let streamSource = null;
let streamProcessor = null;
let realTimeWs = null;
let micStream = null;

function initAudioAnalyser() {
    if (audioCtx) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64;
    
    // Подключаем стандартный плеер для HTTP-фоллбека
    sourceNode = audioCtx.createMediaElementSource(audioPlayer);
    sourceNode.connect(analyser);
    
    analyser.connect(audioCtx.destination);
    frequencyData = new Uint8Array(analyser.frequencyBinCount);
}

// Инициализация WebSocket-соединения с автоподключением (для режима Рации / Текста)
function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
    
    console.log("[WS] Connecting to:", wsUrl);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log("[WS] Connected to chat backend");
        if (modeSelect.value === "walkie-talkie") {
            updateStatus("Готов к общению", "green", false);
        }
    };
    
    ws.onmessage = async (event) => {
        try {
            const data = JSON.parse(event.data);
            await handleWSMessage(data);
        } catch (err) {
            console.error("[WS] Error parsing packet:", err);
        }
    };
    
    ws.onclose = () => {
        console.log("[WS] Disconnected, reconnecting in 2s...");
        if (modeSelect.value === "walkie-talkie") {
            updateStatus("Переподключение...", "yellow", true);
        }
        setTimeout(initWebSocket, 2000);
    };
    
    ws.onerror = (err) => {
        console.error("[WS] Socket error:", err);
    };
}

let mediaRecorder = null;
let audioChunks = [];
let stream = null;

// Настройка распознавания речи (STT) в браузере (для режима Рации)
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = "ru-RU";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isRecording = true;
        recordBtn.classList.add("recording");
        micIcon.className = "fa-solid fa-square"; // Стоп
        hintText.textContent = "Слушаю вас (Браузер)... Говорите.";
        updateStatus("Слушаю голосом", "blue", true);
    };

    recognition.onresult = async (event) => {
        const text = event.results[0][0].transcript;
        if (text) {
            appendMessage("user", text);
            await sendToAI(text);
        }
    };

    recognition.onerror = (event) => {
        console.error("Ошибка STT:", event.error);
        if (event.error === "no-speech") {
            hintText.textContent = "Речь не обнаружена. Попробуйте еще раз.";
        } else {
            hintText.textContent = "Ошибка микрофона. Убедитесь в разрешениях.";
        }
        stopRecordingState();
    };

    recognition.onend = () => {
        stopRecordingState();
    };
} else {
    // Если Web Speech API не поддерживается, выбираем Faster Whisper
    const browserOption = sttSelect.querySelector('option[value="browser"]');
    if (browserOption) {
        browserOption.disabled = true;
    }
    sttSelect.value = "faster-whisper";
}

// Запуск локального рекордера (Faster-Whisper) для режима Рации
async function startLocalRecording() {
    audioChunks = [];
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            isRecording = true;
            recordBtn.classList.add("recording");
            micIcon.className = "fa-solid fa-square"; // Стоп
            hintText.textContent = "Слушаю вас (Локально)... Говорите.";
            updateStatus("Запись звука", "blue", true);
        };

        mediaRecorder.onstop = async () => {
            stopRecordingState();
            
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }

            if (audioChunks.length === 0) {
                hintText.textContent = "Звук не записан. Попробуйте еще раз.";
                return;
            }

            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("file", audioBlob, "recording.webm");

            updateStatus("Распознаю...", "blue", true);
            hintText.textContent = "Разбираю вашу речь с помощью Faster-Whisper...";

            try {
                const response = await fetch("/api/transcribe", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Ошибка транскрибации: ${response.status}`);
                }

                const result = await response.json();
                const text = result.text;

                if (text && text.trim().length > 0) {
                    appendMessage("user", text);
                    await sendToAI(text);
                } else {
                    hintText.textContent = "Не удалось распознать слова. Повторите громче.";
                    updateStatus("Готов к общению", "green", false);
                }
            } catch (error) {
                console.error("Ошибка при локальном STT:", error);
                hintText.textContent = "Ошибка распознавания на сервере. Проверьте логи бэкенда.";
                updateStatus("Готов к общению", "green", false);
            }
        };

        mediaRecorder.start();

    } catch (err) {
        console.error("Ошибка доступа к микрофону:", err);
        hintText.textContent = "Не удалось получить доступ к микрофону: " + err.message;
        stopRecordingState();
    }
}

// Конвертер Float32Array -> PCM 16-bit (для Real-Time)
function float32ToInt16(float32Array) {
    const l = float32Array.length;
    const buf = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return buf.buffer;
}

// Запуск стриминга с микрофона на /ws/phone (Свободные руки)
async function startRealTimeCall() {
    try {
        updateStatus("Подключение...", "yellow", true);
        hintText.textContent = "Устанавливаю соединение с Алисой...";
        
        // Предотвращаем конфликты
        stopSpeaking();
        
        // 1. Инициализируем WebSocket к /ws/phone
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const phoneWsUrl = `${protocol}//${window.location.host}/ws/phone`;
        
        console.log("[Phone WS] Connecting to:", phoneWsUrl);
        realTimeWs = new WebSocket(phoneWsUrl);
        
        realTimeWs.onopen = async () => {
            console.log("[Phone WS] Connected successfully!");
            updateStatus("В эфире (VAD)", "red", true);
            hintText.textContent = "Диалог активен. Говорите прямо сейчас, вас слышат!";
            
            // Отправляем стартовые настройки
            const configPayload = {
                voice: voiceSelect.value,
                speed: parseFloat(speedRange.value),
                steps: parseInt(stepsRange.value),
                backend: brainSelect.value,
                api_url: apiUrlInput.value
            };
            realTimeWs.send(JSON.stringify(configPayload));
            
            // 2. Захватываем микрофон с принудительной передискретизацией в 16кГц!
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            streamCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            streamSource = streamCtx.createMediaStreamSource(micStream);
            
            // Буфер на 2048 отсчетов
            streamProcessor = streamCtx.createScriptProcessor(2048, 1, 1);
            streamSource.connect(streamProcessor);
            streamProcessor.connect(streamCtx.destination);
            
            streamProcessor.onaudioprocess = (e) => {
                if (realTimeWs && realTimeWs.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmData = float32ToInt16(inputData);
                    realTimeWs.send(pcmData); // Стримим сырой PCM 16кГц!
                }
            };
            
            isRecording = true;
            recordBtn.classList.add("recording");
            micIcon.className = "fa-solid fa-phone-slash"; // Кнопка сброса
        };
        
        realTimeWs.onmessage = async (event) => {
            try {
                if (typeof event.data === "string") {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === "interruption") {
                        console.log("[Phone WS] Barge-in! Stopping playback.");
                        stopSpeaking();
                        updateStatus("В эфире (VAD)", "red", true);
                        hintText.textContent = "Вы перебили Алису. Говорите...";
                    }
                    else if (data.type === "status" && data.status === "processing") {
                        updateStatus("Алиса думает...", "blue", true);
                        hintText.textContent = "Алиса размышляет...";
                    }
                    else if (data.type === "recognized") {
                        console.log("[Phone WS] Recognized user speech:", data.text);
                        appendMessage("user", data.text);
                    }
                    else if (data.type === "token") {
                        if (!currentAssistantMessageElement) {
                            currentAssistantMessageElement = createAssistantMessagePlaceholder();
                        }
                        currentAssistantMessageElement.textContent += data.text;
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    else if (data.type === "audio") {
                        const audioBytes = base64ToArrayBuffer(data.audio);
                        const audioBuffer = await decodeAudio(audioBytes);
                        
                        audioQueue.push({ audioBuffer, text: data.text });
                        if (!isPlayingAudio) {
                            playNextInQueue();
                        }
                    }
                    else if (data.type === "done") {
                        serverDone = true;
                        if (audioQueue.length === 0 && !isPlayingAudio) {
                            finishSpeakingRealTime();
                        }
                    }
                }
            } catch (err) {
                console.error("[Phone WS] Error parsing packet:", err);
            }
        };
        
        realTimeWs.onclose = () => {
            console.log("[Phone WS] Connection closed cleanly");
            stopRealTimeCallState();
        };
        
        realTimeWs.onerror = (err) => {
            console.error("[Phone WS] Error:", err);
            stopRealTimeCallState();
        };
        
    } catch (err) {
        console.error("[Phone WS] Access to microphone failed:", err);
        hintText.textContent = "Не удалось включить звонок: " + err.message;
        stopRealTimeCallState();
    }
}

// Завершение реплики в Real-time
function finishSpeakingRealTime() {
    isSpeaking = false;
    stopSpeakingAnimation();
    updateStatus("В эфире (VAD)", "red", true);
    hintText.textContent = "Алиса закончила. Говорите, вас слушают...";
    currentAssistantMessageElement = null;
}

// Остановка звонка Свободные руки
function stopRealTimeCallState() {
    isRecording = false;
    recordBtn.classList.remove("recording");
    micIcon.className = "fa-solid fa-microphone";
    updateStatus("Готов к общению", "green", false);
    hintText.textContent = "Звонок завершен. Нажмите для нового звонка";
    
    if (streamProcessor) {
        try { streamProcessor.disconnect(); } catch(e){}
        streamProcessor = null;
    }
    if (streamSource) {
        try { streamSource.disconnect(); } catch(e){}
        streamSource = null;
    }
    if (streamCtx) {
        try { streamCtx.close(); } catch(e){}
        streamCtx = null;
    }
    if (micStream) {
        try { micStream.getTracks().forEach(track => track.stop()); } catch(e){}
        micStream = null;
    }
    if (realTimeWs) {
        if (realTimeWs.readyState === WebSocket.OPEN || realTimeWs.readyState === WebSocket.CONNECTING) {
            realTimeWs.close();
        }
        realTimeWs = null;
    }
    stopSpeaking();
}

// Управление записью / звонком при клике на микрофон
recordBtn.addEventListener("click", () => {
    // В режиме real-time кнопка работает как телефонная трубка
    if (modeSelect.value === "real-time") {
        if (isRecording) {
            stopRealTimeCallState();
        } else {
            startRealTimeCall();
        }
        return;
    }

    // Если ИИ сейчас говорит, останавливаем воспроизведение (только для walkie-talkie)
    if (isSpeaking) {
        stopSpeaking();
        updateStatus("Готов к общению", "green", false);
        return;
    }

    if (isRecording) {
        if (sttSelect.value === "browser" && recognition) {
            recognition.stop();
        } else if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
    } else {
        initAudioAnalyser();
        if (audioCtx && audioCtx.state === "suspended") {
            audioCtx.resume();
        }
        
        if (sttSelect.value === "browser" && recognition) {
            recognition.start();
        } else {
            startLocalRecording();
        }
    }
});

// Завершение сессий при смене режима работы
modeSelect.addEventListener("change", () => {
    if (isRecording) {
        if (modeSelect.value === "real-time") {
            stopRecordingState();
        } else {
            stopRealTimeCallState();
        }
    } else {
        stopSpeaking();
    }
});

function stopRecordingState() {
    isRecording = false;
    recordBtn.classList.remove("recording");
    micIcon.className = "fa-solid fa-microphone";
    updateStatus("Готов к общению", "green", false);
}

// Функция-конвертер Base64 в ArrayBuffer
function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

// Обертка для декодирования аудиоданных
function decodeAudio(audioBytes) {
    return new Promise((resolve, reject) => {
        if (!audioCtx) {
            initAudioAnalyser();
        }
        audioCtx.decodeAudioData(audioBytes, resolve, reject);
    });
}

// Обработка сообщений из WebSocket (для режима Рации / Текста)
async function handleWSMessage(data) {
    if (data.type === "token") {
        if (!currentAssistantMessageElement) {
            currentAssistantMessageElement = createAssistantMessagePlaceholder();
        }
        currentAssistantMessageElement.textContent += data.text;
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } 
    else if (data.type === "audio") {
        try {
            const audioBytes = base64ToArrayBuffer(data.audio);
            const audioBuffer = await decodeAudio(audioBytes);
            
            audioQueue.push({ audioBuffer, text: data.text });
            if (!isPlayingAudio) {
                playNextInQueue();
            }
        } catch (err) {
            console.error("[WS] Decode/play error:", err);
        }
    } 
    else if (data.type === "done") {
        serverDone = true;
        console.log("[WS] Text streaming complete");
        if (audioQueue.length === 0 && !isPlayingAudio) {
            finishSpeaking();
        }
    }
}

// Проигрывание следующего элемента из аудио-очереди
function playNextInQueue() {
    if (audioQueue.length === 0) {
        isPlayingAudio = false;
        if (serverDone) {
            if (modeSelect.value === "real-time") {
                finishSpeakingRealTime();
            } else {
                finishSpeaking();
            }
        }
        return;
    }
    
    isPlayingAudio = true;
    isSpeaking = true;
    updateStatus("Говорю", "red", true);
    hintText.textContent = "Алиса говорит...";
    
    const item = audioQueue.shift();
    
    initAudioAnalyser();
    if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    
    const source = audioCtx.createBufferSource();
    source.buffer = item.audioBuffer;
    
    source.connect(analyser);
    
    source.onended = () => {
        currentSourceNode = null;
        playNextInQueue();
    };
    
    currentSourceNode = source;
    source.start(0);
}

// Завершение говорения ассистента (режим Рации)
function finishSpeaking() {
    isSpeaking = false;
    stopSpeakingAnimation();
    updateStatus("Готов к общению", "green", false);
    hintText.textContent = "Нажмите на микрофон, чтобы продолжить";
    currentAssistantMessageElement = null;
}

// Принудительная остановка воспроизведения
function stopSpeaking() {
    if (currentSourceNode) {
        try {
            currentSourceNode.stop();
        } catch (e) {}
        currentSourceNode = null;
    }
    audioPlayer.pause();
    
    audioQueue = [];
    isPlayingAudio = false;
    isSpeaking = false;
    stopSpeakingAnimation();
    currentAssistantMessageElement = null;
}

// Создание сообщения-заглушки для ИИ в чате
function createAssistantMessagePlaceholder() {
    const msg = document.createElement("div");
    msg.className = `message assistant`;
    
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = "";
    
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return bubble;
}

// Основная функция отправки запроса ИИ (режим Рации)
async function sendToAI(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.warn("[WS] Socket is not open. Falling back to HTTP REST...");
        await sendToAIHttp(text);
        return;
    }
    
    stopSpeaking();
    
    updateStatus("Думаю...", "blue", true);
    hintText.textContent = "Алиса размышляет...";
    serverDone = false;
    
    const payload = {
        text: text,
        voice: voiceSelect.value,
        speed: parseFloat(speedRange.value),
        steps: parseInt(stepsRange.value),
        backend: brainSelect.value,
        api_url: apiUrlInput.value
    };
    
    ws.send(JSON.stringify(payload));
}

// Резервная REST HTTP функция отправки запроса
async function sendToAIHttp(text) {
    stopSpeaking();
    
    updateStatus("Думаю...", "blue", true);
    hintText.textContent = "Алиса размышляет...";
    
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                voice: voiceSelect.value,
                speed: parseFloat(speedRange.value),
                steps: parseInt(stepsRange.value),
                backend: brainSelect.value,
                api_url: apiUrlInput.value
            })
        });

        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status}`);
        }

        const responseTextRaw = response.headers.get("X-Response-Text");
        let aiText = "Аудио сгенерировано, но текст ответа отсутствует.";
        if (responseTextRaw) {
            aiText = decodeURIComponent(escape(responseTextRaw));
        }

        appendMessage("assistant", aiText);

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        audioPlayer.src = audioUrl;
        isSpeaking = true;
        updateStatus("Говорю", "red", true);
        hintText.textContent = "Алиса говорит...";
        
        audioPlayer.play();
        startSpeakingAnimation();

        audioPlayer.onended = () => {
            isSpeaking = false;
            stopSpeakingAnimation();
            updateStatus("Готов к общению", "green", false);
            hintText.textContent = "Нажмите на микрофон, чтобы продолжить";
        };

    } catch (error) {
        console.error("Ошибка при запросе к ИИ:", error);
        appendMessage("assistant", "Произошла ошибка при получении ответа от ИИ. Проверьте запущен ли сервер или LM Studio.");
        updateStatus("Готов к общению", "green", false);
        hintText.textContent = "Попробуйте еще раз";
    }
}

// Обновление строки статуса
function updateStatus(text, color, isPulse) {
    statusText.textContent = text;
    statusDot.className = `status-dot ${color}`;
    if (isPulse) {
        statusDot.classList.add("pulse");
    } else {
        statusDot.classList.remove("pulse");
    }
}

// Добавление сообщений в чат-контейнер
function appendMessage(sender, text) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = sender === "assistant" ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';
    
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatContainer.appendChild(msg);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Визуализатор звуковых волн на Canvas
let wavePhase = 0;
function drawWaves() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const baseRadius = 85; 
    
    if (isSpeaking && analyser) {
        analyser.getByteFrequencyData(frequencyData);
    }
    
    wavePhase += 0.05;
    
    const layers = 3;
    const colors = [
        "rgba(138, 43, 226, 0.4)",  
        "rgba(0, 210, 255, 0.35)",  
        "rgba(255, 0, 127, 0.25)"   
    ];
    
    for (let l = 0; l < layers; l++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[l];
        ctx.lineWidth = 3 - l * 0.5;
        
        ctx.shadowBlur = 15;
        ctx.shadowColor = colors[l];
        
        const numPoints = 80;
        for (let i = 0; i <= numPoints; i++) {
            const angle = (i / numPoints) * Math.PI * 2;
            
            let offset = 0;
            if (isSpeaking && frequencyData.length > 0) {
                const freqIndex = Math.floor((i % (numPoints / 2)) / (numPoints / 2) * frequencyData.length);
                const amplitude = frequencyData[freqIndex] / 255;
                offset = Math.sin(angle * 6 + wavePhase * (l + 1)) * 25 * amplitude;
            } else if (isRecording) {
                offset = Math.sin(angle * 8 + wavePhase * 2) * 8;
            } else {
                offset = Math.sin(angle * 4 + wavePhase) * 3;
            }
            
            const r = baseRadius + offset + l * 8;
            const x = centerX + Math.cos(angle) * r;
            const y = centerY + Math.sin(angle) * r;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.closePath();
        ctx.stroke();
    }
    
    ctx.shadowBlur = 0;
    animationFrameId = requestAnimationFrame(drawWaves);
}

function startSpeakingAnimation() {}
function stopSpeakingAnimation() {}

// Запуск анимации
drawWaves();

// Инициализация вебсокета при старте
initWebSocket();

// Обработчик отправки текстовых сообщений
chatInputForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = chatTextInput.value.trim();
    if (!text) return;
    
    chatTextInput.value = "";
    
    if (isSpeaking) {
        stopSpeaking();
    }
    
    if (isRecording) {
        if (modeSelect.value === "real-time") {
            stopRealTimeCallState();
        } else if (sttSelect.value === "browser" && recognition) {
            recognition.stop();
        } else if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
    }
    
    initAudioAnalyser();
    if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    
    appendMessage("user", text);
    await sendToAI(text);
});
