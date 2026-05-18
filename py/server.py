import os
import io
import json
import uuid
import httpx
import uvicorn
import asyncio
import base64
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from supertonic import TTS
from ru_preprocessor import RussianTTSPreprocessor
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

app = FastAPI(title="Supertonic TTS & Live LLM Voice Assistant")

# Инициализируем наш препроцессор ударений
preprocessor = RussianTTSPreprocessor()

# Инициализируем TTS-движок Supertonic (в оффлайн-режиме, так как модели уже скачаны)
print("[TTS] Loading Supertonic model...")
tts = TTS(auto_download=False)

# Инициализируем STT-модель Faster-Whisper ("tiny" для ультравысокой скорости на CPU)
print("[STT] Loading Faster-Whisper 'tiny' model on CPU (int8)...")
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Инициализируем Silero VAD для телефонии и WebRTC
print("[VAD] Loading Silero VAD model...")
vad_model = load_silero_vad()

# Пул потоков для выполнения тяжелого CPU-синтеза TTS, чтобы не блокировать Event Loop
executor = ThreadPoolExecutor(max_workers=4)

# Слайды голосов
VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

class ChatRequest(BaseModel):
    text: str
    voice: str = "F2"
    speed: float = 1.15
    steps: int = 12
    backend: str = "lm-studio" # vllm, lm-studio, ollama, gemini, mock
    api_url: str = None

# Вспомогательная функция для извлечения полных предложений из потока
def extract_sentences(buffer: str, is_final: bool = False):
    sentences = []
    punctuation_marks = {'.', '!', '?', '\n'}
    i = 0
    last_cut = 0
    while i < len(buffer):
        char = buffer[i]
        if char in punctuation_marks:
            while i + 1 < len(buffer) and buffer[i + 1] in punctuation_marks:
                i += 1
            preceding_part = buffer[last_cut:i+1].strip()
            is_abbrev = False
            for abbrev in ["т.д.", "т.п.", "др.", "руб.", "тыс.", "млн."]:
                if preceding_part.lower().endswith(abbrev):
                    is_abbrev = True
                    break
            
            if not is_abbrev:
                sentence = buffer[last_cut:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                last_cut = i + 1
        i += 1
        
    if is_final:
        sentence = buffer[last_cut:].strip()
        if sentence:
            sentences.append(sentence)
        remaining = ""
    else:
        remaining = buffer[last_cut:]
        
    return sentences, remaining

# Стриминг из OpenAI-совместимых бэкендов (vLLM, LM Studio)
async def stream_from_openai(url: str, messages: list, model: str = "local-model") -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST",
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "stream": True
                },
                timeout=15.0
            ) as response:
                if response.status_code != 200:
                    print(f"[LLM Stream] Server returned status {response.status_code}")
                    return
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            pass
        except Exception as e:
            print(f"[LLM Stream] Connection error: {e}")

# Вспомогательная функция для получения быстрой и доступной модели в Ollama
async def get_active_ollama_model(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(f"{url}/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if models:
                # Ищем легкие модели в приоритете
                for m in models:
                    m_lower = m.lower()
                    if any(fast in m_lower for fast in ["0.5b", "1.5b", "qwen", "phi", "mini"]):
                        print(f"[Ollama] Autodetected super-fast model: {m}")
                        return m
                # Иначе берем первую доступную
                print(f"[Ollama] Selected first available model: {models[0]}")
                return models[0]
    except Exception as e:
        print(f"[Ollama] Failed to fetch active models: {e}")
    return "llama3"

# Стриминг из Ollama с оптимизацией скорости
async def stream_from_ollama(url: str, prompt: str) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient() as client:
        model_name = await get_active_ollama_model(client, url)
        try:
            async with client.stream(
                "POST",
                f"{url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 45,       # Жесткий лимит токенов для сверхбыстрого Real-Time
                        "num_ctx": 1024,         # Меньше контекст -> мгновенный старт
                        "temperature": 0.5,      # Сниженная температура для стабильности и скорости
                        "top_k": 10,
                        "top_p": 0.5,
                        "num_thread": 4,         # Принудительное использование 4 ядер CPU для скорости
                        "mirostat": 0            # Отключение сложного сэмплирования для ускорения
                    }
                },
                timeout=15.0
            ) as response:
                if response.status_code != 200:
                    print(f"[Ollama Stream] Server returned status {response.status_code}")
                    return
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Ollama Stream] Connection error: {e}")

# Стриминг из Gemini
async def stream_from_gemini(system_prompt: str, prompt: str) -> AsyncGenerator[str, None]:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("[Gemini Stream] API Key not found")
        return
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(f"{system_prompt}\n\nПользователь: {prompt}", stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"[Gemini Stream] Error: {e}")

# Стриминг из заглушки
async def stream_mock(text: str) -> AsyncGenerator[str, None]:
    witty_responses = {
        "привет": "Привет! Рада слышать тебя. О чем сегодня поговорим?",
        "как дела": "У меня всё отлично, летаю на антигравитации! А как твои успехи?",
        "кто ты": "Я твой голосовой собеседник, озвученный моделью Supertonic.",
        "велосипед": "О, велосипед — это круто! Помню, мы недавно обсуждали, как держать тридцать километров в час.",
    }
    user_lower = text.lower()
    ai_text = "Я услышала тебя! Давай пообщаемся о чем-нибудь интересном сегодня."
    for key, val in witty_responses.items():
        if key in user_lower:
            ai_text = val
            break
    words = ai_text.split(" ")
    for word in words:
        yield word + " "
        await asyncio.sleep(0.08)

# WebSocket-эндпоинт для настоящей телефонии (SIP / WebRTC) со встроенным Silero VAD и Barge-in
@app.websocket("/ws/phone")
async def websocket_phone(websocket: WebSocket):
    await websocket.accept()
    print("[Phone WS] Caller connected")
    
    # Дефолтные настройки для телефонии (потоковость и скорость критически важны)
    voice = "F2"
    speed = 1.15
    steps = 6  # 6 шагов Supertonic дают идеальное качество и снижают время синтеза на 25%
    backend = "gemini" # vllm, lm-studio, ollama, gemini, mock
    api_url = None
    
    # Инициализируем VADIterator для частоты 16кГц (размер кадра 512 отсчетов)
    vad_iterator = VADIterator(
        vad_model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=250, # 250 мс тишины - признак конца реплики (выигрыш 350 мс!)
        speech_pad_ms=30
    )
    
    # Буферы для звука
    byte_buffer = b""
    speech_chunks = []
    
    # Флаги состояний
    is_speaking = False
    speech_task = None
    
    # Системный промпт для телефона (сверхкратко и динамично для мгновенного первого токена)
    system_prompt = (
        "Ты - дружелюбный голосовой ассистент по имени Алиса. Отвечай КРАЙНЕ КРАТКО (строго одна фраза, "
        "не более 12 слов), живо, весело и естественно на русском языке. Используй простую разговорную речь. Никаких списков."
    )
    
    # Асинхронная функция генерации и отправки ответа
    async def run_voice_pipeline(user_text: str):
        nonlocal speech_task
        try:
            print(f"[Phone LLM] User text: {user_text}")
            # Определяем источник стриминга
            stream_gen = None
            if backend == "vllm":
                url = api_url.strip() if api_url else "http://localhost:8000/v1"
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
                stream_gen = stream_from_openai(url, messages, model="local-model")
            elif backend == "lm-studio":
                url = api_url.strip() if api_url else "http://localhost:1234/v1"
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
                stream_gen = stream_from_openai(url, messages, model="local-model")
            elif backend == "ollama":
                url = api_url.strip() if api_url else "http://localhost:11434"
                prompt = f"{system_prompt}\n\nПользователь: {user_text}\nАссистент:"
                stream_gen = stream_from_ollama(url, prompt)
            elif backend == "gemini":
                stream_gen = stream_from_gemini(system_prompt, user_text)
            else:
                stream_gen = stream_mock(user_text)
                
            buffer = ""
            voice_style = tts.get_voice_style(voice_name=voice)
            loop = asyncio.get_running_loop()
            
            async for token in stream_gen:
                buffer += token
                # Отправляем текстовый токен для логов/дебага
                await websocket.send_json({"type": "token", "text": token})
                
                # Пробуем выделить предложения
                sentences, buffer = extract_sentences(buffer, is_final=False)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if not sentence_clean:
                        continue
                        
                    processed_text = preprocessor.apply_rules(sentence_clean)
                    print(f"[Phone TTS] Synthesizing: {processed_text}")
                    
                    # Синтез в пуле потоков
                    wav, _ = await loop.run_in_executor(
                        executor,
                        lambda: tts.synthesize(
                            text=processed_text,
                            lang="ru",
                            voice_style=voice_style,
                            total_steps=steps,
                            speed=speed
                        )
                    )
                    
                    temp_file = f"temp_{uuid.uuid4()}.wav"
                    try:
                        tts.save_audio(wav, temp_file)
                        with open(temp_file, "rb") as f:
                            wav_bytes = f.read()
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    await websocket.send_json({
                        "type": "audio",
                        "text": sentence_clean,
                        "audio": audio_b64
                    })
                    
            if buffer.strip():
                sentences, _ = extract_sentences(buffer, is_final=True)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if not sentence_clean:
                        continue
                    processed_text = preprocessor.apply_rules(sentence_clean)
                    print(f"[Phone TTS Final] Synthesizing: {processed_text}")
                    
                    wav, _ = await loop.run_in_executor(
                        executor,
                        lambda: tts.synthesize(
                            text=processed_text,
                            lang="ru",
                            voice_style=voice_style,
                            total_steps=steps,
                            speed=speed
                        )
                    )
                    
                    temp_file = f"temp_{uuid.uuid4()}.wav"
                    try:
                        tts.save_audio(wav, temp_file)
                        with open(temp_file, "rb") as f:
                            wav_bytes = f.read()
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    await websocket.send_json({
                        "type": "audio",
                        "text": sentence_clean,
                        "audio": audio_b64
                    })
            await websocket.send_json({"type": "done"})
        except asyncio.CancelledError:
            print("[Phone Task] Pipeline was cancelled due to user interruption (Barge-in)!")
        except Exception as err:
            print(f"[Phone Task Error] {err}")
            
    try:
        while True:
            # WebSocket может прислать как текстовое (JSON с настройками), так и бинарное сообщение (аудио-пакет)
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                print("[Phone WS] Client disconnected cleanly")
                break
            
            # 1. Текстовое сообщение (Настройки бэкенда/голоса от АТС или клиента)
            if "text" in message:
                data_json = message["text"]
                req_data = json.loads(data_json)
                voice = req_data.get("voice", voice)
                speed = float(req_data.get("speed", speed))
                steps = int(req_data.get("steps", steps))
                backend = req_data.get("backend", backend)
                api_url = req_data.get("api_url", api_url)
                print(f"[Phone WS] Applied config: voice={voice}, backend={backend}, speed={speed}")
                
            # 2. Бинарное сообщение (Аудио-пакет от SIP-телефонии или WebRTC-клиента)
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                if not audio_bytes:
                    continue
                
                # Добавляем в буфер
                byte_buffer += audio_bytes
                
                # Размер кадра для VAD на 16кГц: 512 отсчетов * 2 байта (для int16) = 1024 байта
                FRAME_SIZE_BYTES = 1024
                
                while len(byte_buffer) >= FRAME_SIZE_BYTES:
                    frame = byte_buffer[:FRAME_SIZE_BYTES]
                    byte_buffer = byte_buffer[FRAME_SIZE_BYTES:]
                    
                    # Декодируем 16-битный PCM в float32 numpy-массив
                    audio_chunk_int16 = np.frombuffer(frame, dtype=np.int16)
                    audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
                    
                    # Проверяем активность речи
                    speech_dict = vad_iterator(audio_chunk_float32, return_seconds=False)
                    
                    if speech_dict:
                        # А. Обнаружено начало речи! (Пользователь начал говорить)
                        if "start" in speech_dict:
                            print("[Phone VAD] Speech started! User is speaking...")
                            is_speaking = True
                            speech_chunks = []
                            
                            # Прерывание (Barge-in): если робот сейчас говорит, немедленно прерываем его!
                            if speech_task and not speech_task.done():
                                speech_task.cancel()
                                print("[Phone VAD] Barge-in! Speech task cancelled.")
                                # Сигнализируем АТС/клиенту о прерывании, чтобы они очистили аудиобуфер воспроизведения
                                await websocket.send_json({"type": "interruption"})
                        
                        # Б. Обнаружен конец речи! (Пользователь закончил фразу)
                        elif "end" in speech_dict:
                            print("[Phone VAD] Speech ended. Analyzing...")
                            is_speaking = False
                            
                            # Если у нас есть записанные фрагменты голоса
                            if speech_chunks:
                                # Объединяем все чанки в один непрерывный аудио-массив
                                full_audio = np.concatenate(speech_chunks, axis=0)
                                print(f"[Phone VAD] Audio length for STT: {len(full_audio)/16000:.2f} seconds")
                                
                                # Отправляем сигнал "Распознаю"
                                await websocket.send_json({"type": "status", "status": "processing"})
                                
                                # Распознаем речь
                                segments, info = stt_model.transcribe(full_audio, language="ru", beam_size=3)
                                text = "".join(segment.text for segment in segments).strip()
                                
                                if text:
                                    print(f"[Phone User] {text}")
                                    await websocket.send_json({"type": "recognized", "text": text})
                                    
                                    # Запускаем генерацию и озвучку в фоне
                                    speech_task = asyncio.create_task(run_voice_pipeline(text))
                                else:
                                    print("[Phone STT] Speech was empty or not recognized.")
                                    await websocket.send_json({"type": "status", "status": "ready"})
                                    
                            speech_chunks = []
                            
                    # Если пользователь сейчас говорит, накапливаем аудиоданные
                    if is_speaking:
                        speech_chunks.append(audio_chunk_float32.copy())
                        
    except WebSocketDisconnect:
        print("[Phone WS] Caller disconnected")
        if speech_task and not speech_task.done():
            speech_task.cancel()
    except Exception as e:
        print(f"[Phone WS Error] {e}")
        if speech_task and not speech_task.done():
            speech_task.cancel()

# WebSocket endpoint для real-time низкозадержечного общения
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    try:
        while True:
            # Получаем настройки и текст от пользователя
            data_json = await websocket.receive_text()
            req_data = json.loads(data_json)
            
            user_text = req_data.get("text", "").strip()
            voice = req_data.get("voice", "F2")
            speed = float(req_data.get("speed", 1.15))
            steps = int(req_data.get("steps", 6)) # Для стриминга по умолчанию 6 шагов (ультранизкая задержка)
            backend = req_data.get("backend", "lm-studio")
            api_url = req_data.get("api_url", None)
            
            if not user_text:
                continue
                
            print(f"[WS] Received user input: {user_text}")
            
            # Системный промпт для жесткого ограничения длины ответа (сверхбыстрый инференс и синтез)
            system_prompt = (
                "Ты - дружелюбный голосовой ассистент по имени Алиса. Отвечай КРАЙНЕ КРАТКО (строго одна короткая фраза, "
                "не более 12 слов), живо, весело и естественно на русском языке. Используй простую разговорную речь. Никаких списков."
            )
            
            # Определяем источник стриминга
            stream_gen = None
            if backend == "vllm":
                url = api_url.strip() if api_url else "http://localhost:8000/v1"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
                stream_gen = stream_from_openai(url, messages, model="local-model")
            elif backend == "lm-studio":
                url = api_url.strip() if api_url else "http://localhost:1234/v1"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
                stream_gen = stream_from_openai(url, messages, model="local-model")
            elif backend == "ollama":
                url = api_url.strip() if api_url else "http://localhost:11434"
                prompt = f"{system_prompt}\n\nПользователь: {user_text}\nАссистент:"
                stream_gen = stream_from_ollama(url, prompt)
            elif backend == "gemini":
                stream_gen = stream_from_gemini(system_prompt, user_text)
            else:
                stream_gen = stream_mock(user_text)
                
            # Запускаем чтение стрима и разделение на предложения
            buffer = ""
            voice_style = tts.get_voice_style(voice_name=voice if voice in VOICES else "F2")
            loop = asyncio.get_running_loop()
            
            async for token in stream_gen:
                buffer += token
                # Отправляем текстовый токен клиенту для мгновенного отображения на экране
                await websocket.send_json({"type": "token", "text": token})
                
                # Пробуем выделить законченные предложения
                sentences, buffer = extract_sentences(buffer, is_final=False)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if not sentence_clean:
                        continue
                        
                    # Обрабатываем предложение через наш фонетический препроцессор ударений и редукции
                    processed_text = preprocessor.apply_rules(sentence_clean)
                    print(f"[WS Sentence] Synthesizing: {processed_text}")
                    
                    # Запускаем синтез в пуле потоков, чтобы не блокировать асинхронный event loop!
                    try:
                        wav, _ = await loop.run_in_executor(
                            executor,
                            lambda: tts.synthesize(
                                text=processed_text,
                                lang="ru",
                                voice_style=voice_style,
                                total_steps=steps,
                                speed=speed
                            )
                        )
                        
                        # Сохраняем во временный уникальный файл с UUID
                        temp_file = f"temp_{uuid.uuid4()}.wav"
                        try:
                            tts.save_audio(wav, temp_file)
                            with open(temp_file, "rb") as f:
                                wav_bytes = f.read()
                        finally:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                
                        # Кодируем аудио в base64 и отправляем
                        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio",
                            "text": sentence_clean,
                            "audio": audio_b64
                        })
                    except Exception as tts_err:
                        print(f"[WS TTS Error] {tts_err}")
                        
            # Обрабатываем остаток буфера
            if buffer.strip():
                sentences, _ = extract_sentences(buffer, is_final=True)
                for sentence in sentences:
                    sentence_clean = sentence.strip()
                    if not sentence_clean:
                        continue
                    processed_text = preprocessor.apply_rules(sentence_clean)
                    print(f"[WS Final Sentence] Synthesizing: {processed_text}")
                    try:
                        wav, _ = await loop.run_in_executor(
                            executor,
                            lambda: tts.synthesize(
                                text=processed_text,
                                lang="ru",
                                voice_style=voice_style,
                                total_steps=steps,
                                speed=speed
                            )
                        )
                        
                        temp_file = f"temp_{uuid.uuid4()}.wav"
                        try:
                            tts.save_audio(wav, temp_file)
                            with open(temp_file, "rb") as f:
                                wav_bytes = f.read()
                        finally:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                
                        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio",
                            "text": sentence_clean,
                            "audio": audio_b64
                        })
                    except Exception as tts_err:
                        print(f"[WS TTS Error] {tts_err}")
                        
            # Отправляем сигнал завершения текущего ответа
            await websocket.send_json({"type": "done"})
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS Error] {e}")

# Оставляем существующий REST API /api/chat для совместимости
@app.post("/api/chat")
async def chat_and_synthesize(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    
    ai_text = ""
    system_prompt = (
        "Ты - дружелюбный голосовой ассистент по имени Алиса. Отвечай кратко (1-3 предложения), "
        "живо, вежливо и естественно на русском языке. Используй простую разговорную речь."
    )
    
    if req.backend == "vllm":
        url = req.api_url.strip() if req.api_url else "http://localhost:8000/v1"
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{url}/chat/completions",
                    json={
                        "model": "local-model",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": req.text}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 150
                    },
                    timeout=10.0
                )
                if res.status_code == 200:
                    ai_text = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[vLLM] returned status {res.status_code}")
        except Exception as e:
            print(f"[vLLM] Connection error")

    if not ai_text and req.backend == "lm-studio":
        url = req.api_url.strip() if req.api_url else "http://localhost:1234/v1"
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{url}/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": req.text}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 150
                    },
                    timeout=10.0
                )
                if res.status_code == 200:
                    ai_text = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    print(f"[LM Studio] returned status {res.status_code}")
        except Exception as e:
            print(f"[LM Studio] Connection error")
            
    # Резервный бэкенд: Ollama
    if not ai_text and (req.backend == "ollama" or req.backend == "lm-studio"):
        url = req.api_url.strip() if req.api_url else "http://localhost:11434"
        try:
            async with httpx.AsyncClient() as client:
                model_name = await get_active_ollama_model(client, url)
                res = await client.post(
                    f"{url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": f"{system_prompt}\n\nПользователь: {req.text}\nАссистент:",
                        "stream": False,
                        "options": {
                            "num_predict": 45,
                            "num_ctx": 1024,
                            "temperature": 0.7,
                            "top_k": 20,
                            "top_p": 0.9
                        }
                    },
                    timeout=10.0
                )
                if res.status_code == 200:
                    ai_text = res.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama] Connection error: {e}")

    # Резервный бэкенд: Gemini API
    if not ai_text and req.backend == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"{system_prompt}\n\nПользователь: {req.text}")
                ai_text = response.text
            except Exception as e:
                print(f"[Gemini] API error")

    # Полный фоллбэк (если нет сети или запущенных локальных ИИ)
    if not ai_text:
        witty_responses = {
            "привет": "Привет! Рада слышать тебя. О чем сегодня поговорим?",
            "как дела": "У меня всё отлично, летаю на антигравитации! А как твои успехи?",
            "кто ты": "Я твой голосовой собеседник, озвученный моделью Supertonic.",
            "велосипед": "О, велосипед — это круто! Помню, мы недавно обсуждали, как держать тридцать километров в час.",
        }
        user_lower = req.text.lower()
        for key, val in witty_responses.items():
            if key in user_lower:
                ai_text = val
                break
        if not ai_text:
            ai_text = "Я услышала тебя, но сейчас у меня нет связи с локальным ИИ в LM Studio или Ollama. Проверь, запущены ли они!"

    print(f"[AI] Response (safe log): {ai_text.encode('ascii', errors='backslashreplace').decode('ascii')}")
    
    # 2. Нормализация текста препроцессором (правильные ударения)
    processed_text = preprocessor.apply_rules(ai_text)
    print(f"[Normalized] Text (safe log): {processed_text.encode('ascii', errors='backslashreplace').decode('ascii')}")
    
    # 3. Синтез через Supertonic TTS
    try:
        voice_style = tts.get_voice_style(voice_name=req.voice if req.voice in VOICES else "F2")
        wav, _ = await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: tts.synthesize(
                text=processed_text,
                lang="ru",
                voice_style=voice_style,
                total_steps=req.steps,
                speed=req.speed
            )
        )
        
        # Сохраняем во временный WAV файл
        temp_file = "temp_response.wav"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
        tts.save_audio(wav, temp_file)
        
        return FileResponse(
            temp_file,
            media_type="audio/wav",
            headers={
                "X-Response-Text": ai_text.encode("utf-8").decode("latin1") # Обходим ограничение HTTP-заголовков на не-ascii
            }
        )
        
    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed")

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Читаем содержимое файла
        audio_bytes = await file.read()
        
        # Сохраняем во временный файл
        temp_filename = "temp_recording.webm"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
            
        # Транскрибируем с помощью faster-whisper
        # Beam size = 5 дает отличное качество, language="ru" форсирует русский язык
        segments, info = stt_model.transcribe(temp_filename, language="ru", beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
        
        # Удаляем временный файл
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass
            
        print(f"[STT] Transcribed text: {text}")
        return {"text": text}
    except Exception as e:
        print(f"[STT] Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Монтируем статические файлы для фронтенда
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
