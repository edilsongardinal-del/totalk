# -*- coding: utf-8 -*-
## test_call_real.py — Pipeline robusto de chamada com STT/TTS funcional
import asyncio
import json
import logging
import os
import time
import uuid
import base64
import audioop
import hashlib
from typing import Dict, Any, List, Optional
from asyncio import Queue
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import io
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.websockets import WebSocketState
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
from pydub import AudioSegment
from google.cloud import speech_v1
import contextlib
from contextlib import asynccontextmanager, contextmanager, suppress

# Importações locais
from agent_intelligence import LiviaTurbinadaAgent
from improved_orchestration import ImprovedOrchestrator, ConversationContext, enhanced_generate_and_speak
from regras import (
    sanitize_speech,
    robust_extract_email,
    is_likely_name,
    extract_day_hint_enhanced,
    normalize_contact_time_phrase_enhanced,
    validate_email_with_retry,
    make_email_speakable,
    rules_handler_func,
    get_context_reengagement_message
)
from human_nuance import render_natural, affect_from_text, voice_settings_for_affect

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
log = logger

# --- Constantes ---
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8004))
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID")
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER")
PUBLIC_HTTP_URL = os.getenv("PUBLIC_HTTP_URL", f"http://localhost:{APP_PORT}")
PUBLIC_WSS_URL = os.getenv("PUBLIC_WSS_URL", f"ws://localhost:{APP_PORT}/test-ws-stream")
TELNYX_BASE_URL = "https://api.telnyx.com/v2"
STREAM_CODEC = "PCMU"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_TTS_MODEL_LONG = os.getenv("ELEVENLABS_TTS_MODEL_LONG", "eleven_multilingual_v2")

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Parâmetros de áudio
SAMPLES_PER_FRAME_8K = 160
FRAME_MS = 20
SILENCE_RMS = int(os.getenv("SILENCE_RMS", "200"))
SILENCE_GAP_S = float(os.getenv("SILENCE_GAP_S", "0.8"))
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "300"))
MAX_UTTER_MS = float(os.getenv("MAX_UTTER_MS", "10000"))
BARGE_MS = 500
REENGAGEMENT_TIMEOUT_S = 10.0
MAX_REENGAGEMENTS = 2
SHORT_FRAGMENT_CHARS = 20
MERGE_WINDOW_MS = 1500

# --- Estado da chamada ---
@dataclass
class CallState:
    call_id: str
    ws: WebSocket
    context: Optional[ConversationContext] = None 
    ws_open: bool = True
    is_processing_user_input: bool = False
    codec_in: str = "PCMU"
    codec_out: str = "PCMU"
    is_generating_tts: bool = False
    is_bot_speaking: bool = False
    tts_cancel: asyncio.Event = field(default_factory=asyncio.Event)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_bot_speech_ended_at: float = 0.0
    voice_ms_during_tts: float = 0.0
    bot_phrase_started_at: Optional[float] = None
    last_barge_cancel_ts: float = 0.0
    awaiting_answer: bool = False
    total_transcriptions: int = 0
    empty_transcriptions: int = 0
    conversation_ended: bool = False   
    opening_sequence_done: bool = False  
    bot_speech_ended_at: float = 0.0    
    _recent_user_chars: List[tuple] = field(default_factory=list, init=False, repr=False)
    pending_user_text: Optional[str] = None
    pending_since: float = 0.0
    last_reengagement_at = 0.0
    reengage_inflight = False


    def add_message(self, role: str, text: str):
        if not text or not text.strip(): return
        message_content = f"{role}:{text.strip()}"
        message_hash = hashlib.md5(message_content.encode()).hexdigest()
        if message_hash in [hashlib.md5(f"{m['role']}:{m['text']}".encode()).hexdigest() for m in self.conversation_history]:
            return
        self.conversation_history.append({"role": role, "text": text})
        log.debug(f"[HISTORY] {role.upper()}: {text[:50]}...")

    def barge_refractory(self, seconds: float = 1.2) -> bool:
        return (time.monotonic() - self.last_barge_cancel_ts) < seconds

    def reset(self):
        self.tts_cancel.set()
        self.ws_open = False

    def note_asr_len(self, n: int):
        self._recent_user_chars.append((time.monotonic(), n))
        if len(self._recent_user_chars) > 10:
            self._recent_user_chars.pop(0)

# --- Globais ---
active_calls: Dict[str, CallState] = {}
tts_queue: Optional[Queue] = None
client_llm: Optional[httpx.AsyncClient] = None
client_el: Optional[httpx.AsyncClient] = None
client_openai: Optional[httpx.AsyncClient] = None
_speech_client = None

# --- Google STT ---

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    log.warning("GOOGLE_APPLICATION_CREDENTIALS não configurado. STT pode falhar.")

def get_speech_client():
    global _speech_client
    if _speech_client is None:        
        _speech_client = speech_v1.SpeechClient()
    return _speech_client

def google_transcribe_pcm16_8k(pcm16_8k: bytes) -> str:
    client = get_speech_client()
    audio = speech_v1.RecognitionAudio(content=pcm16_8k)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="pt-BR",
        enable_automatic_punctuation=True,
        model="telephony",
        speech_contexts=[speech_v1.SpeechContext(phrases=[
            "gmail", "hotmail", "arroba", "ponto com", "seguro de vida", "Livia", "Cardinalle"
        ])]
    )
    try:
        resp = client.recognize(config=config, audio=audio)
        if resp.results and resp.results[0].alternatives:
            return resp.results[0].alternatives[0].transcript.strip()
    except Exception as e:
        log.error(f"[STT] Erro: {e}")
    return ""

# --- Áudio ---
def pcm16_to_pcmu(pcm16_bytes: bytes) -> bytes:
    return audioop.lin2ulaw(pcm16_bytes, 2) if pcm16_bytes else b""

def pcm16_to_pcma(pcm16_bytes: bytes) -> bytes:
    return audioop.lin2alaw(pcm16_bytes, 2) if pcm16_bytes else b""

def mp3_to_pcm16_8k(mp3_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    return audio.raw_data

# --- TTS ---
async def send_audio_frames(websocket: WebSocket, encoded_bytes: bytes, call_state: CallState):
    if not websocket or not call_state.ws_open or not encoded_bytes: return
    frame_size = SAMPLES_PER_FRAME_8K
    codec = (call_state.codec_out or STREAM_CODEC).upper()
    pad_byte = b"\xD5" if codec == "PCMA" else b"\xFF"
    next_tp = time.monotonic()
    period = FRAME_MS / 1000.0
    for i in range(0, len(encoded_bytes), frame_size):
        if call_state.tts_cancel.is_set() or not call_state.ws_open: break
        chunk = encoded_bytes[i:i+frame_size]
        if len(chunk) < frame_size:
            chunk += pad_byte * (frame_size - len(chunk))
        pkt = {"event": "media", "media": {"track": "outbound", "payload": base64.b64encode(chunk).decode("ascii")}}
        try:
            await websocket.send_text(json.dumps(pkt))
        except Exception as e:
            log.error(f"[WS] Erro ao enviar frame: {e}")
            break
        next_tp += period
        delay = next_tp - time.monotonic()
        if delay > 0: await asyncio.sleep(delay)
        else: next_tp = time.monotonic()

async def speak_response(call_state: CallState, text: str, voice_settings: Dict[str, Any], mode: str = "live"):
    if not call_state or not call_state.ws_open:
        return

    text = sanitize_speech(text)
    if not text.strip():
        return
    
    call_state.is_bot_speaking = True
    call_state.awaiting_answer = False
    call_state.bot_phrase_started_at = time.monotonic()
 
    await tts_queue.put({
        "text": text,
        "voice_settings": voice_settings,
        "mode": mode,
        "ws": call_state.ws,
        "call_state": call_state,
    })


async def _tts_processor():
    global tts_queue
    while True:
        item = await tts_queue.get()
        call_state = None
        try:
            if item is None:
                break

            text = item.get("text", "")
            voice_settings = item.get("voice_settings", {})
            mode = item.get("mode", "live")
            ws = item.get("ws")
            call_state = item.get("call_state")

            # WS/estado válidos?
            if not ws or not call_state or not call_state.ws_open:
                log.warning(f"WebSocket fechado, descartando TTS: '{text[:30]}...'")
                continue
            if hasattr(ws, "client_state") and ws.client_state != WebSocketState.CONNECTED:
                log.warning(f"WebSocket não conectado, descartando TTS: '{text[:30]}...'")
                continue

            codec_out = call_state.codec_out if call_state else STREAM_CODEC

            # --- Gera bytes de áudio já no codec da Telnyx (A-law/μ-law a 8 kHz) ---
            if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
                # fallback: 20 ms de silêncio
                silence_byte = b'\xff' if codec_out == "PCMU" else b'\xd5'
                audio_bytes = silence_byte * 160
            else:
                payload = {
                    "text": text,
                    "model_id": ELEVENLABS_TTS_MODEL,
                    "voice_settings": voice_settings,
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                        headers={"xi-api-key": ELEVENLABS_API_KEY},
                        json=payload,
                    )
                if resp.status_code != 200:
                    log.error("Falha ElevenLabs: %s", resp.text)
                    silence_byte = b'\xff' if codec_out == "PCMU" else b'\xd5'
                    audio_bytes = silence_byte * 160
                else:
                    audio_bytes_mp3 = resp.content
                    audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes_mp3))
                    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    pcm16_bytes = audio.raw_data
                    if codec_out == "PCMA":
                        audio_bytes = audioop.lin2alaw(pcm16_bytes, 2)
                    else:
                        audio_bytes = audioop.lin2ulaw(pcm16_bytes, 2)

            # --- Envia em frames de 20ms (160 amostras @ 8kHz mono) ---
            frame_size = 160
            for i in range(0, len(audio_bytes), frame_size):
                # Se houver lógica de "barge-in", interrompa aqui:
                if getattr(call_state, "cancel_tts", False):
                    log.info("TTS cancelado por barge-in.")
                    break

                chunk = audio_bytes[i:i+frame_size]
                if not chunk:
                    continue
                payload_b64 = base64.b64encode(chunk).decode()

                try:
                    await ws.send_json({
                        "event": "media",
                        "stream_id": call_state.call_id,
                        "media": {
                            "payload": payload_b64,
                            "codec": codec_out,
                        },
                    })
                except RuntimeError as e:
                    if "close message has been sent" in str(e):
                        log.warning(f"WebSocket fechado durante envio de TTS: '{text[:30]}...'")
                        break
                    raise

                # Ritmo ~ tempo real (20ms por frame)
                await asyncio.sleep(0.02)

            log.info(f"TTS processado e enviado: '{text[:50]}...'")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Erro no _tts_processor: {e}", exc_info=True)
        finally:
            # Finaliza o item da fila e reseta estado de fala do bot
            try:
                if call_state:
                    call_state.is_bot_speaking = False
                    call_state.bot_speech_ended_at = time.monotonic()
                    call_state.awaiting_answer = True
            except Exception:
                pass
            tts_queue.task_done()


# --- STT Processor (VAD + Transcrição) ---
async def stt_processor_google(audio_queue: Queue, call_state: CallState, orchestrator: ImprovedOrchestrator):
    speech_buffer = bytearray()
    last_voice_ts = 0
    SILENCE_GAP_S_LOCAL = 0.8
    MIN_SPEECH_S = 0.3
    while call_state.ws_open:
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=SILENCE_GAP_S_LOCAL / 2)
            if chunk is None: break
            rms = audioop.rms(chunk, 2)
            is_speech = rms > SILENCE_RMS
            if is_speech:
                speech_buffer.extend(chunk)
                last_voice_ts = time.monotonic()
            elif speech_buffer:
                if (time.monotonic() - last_voice_ts) > SILENCE_GAP_S_LOCAL:
                    duration_s = len(speech_buffer) / (8000 * 2)
                    if duration_s >= MIN_SPEECH_S:
                        transcript = await asyncio.get_running_loop().run_in_executor(None, google_transcribe_pcm16_8k, bytes(speech_buffer))
                        call_state.total_transcriptions += 1
                        if transcript:
                            log.info(f"Transcrição final: '{transcript}'")
                            call_state.add_message("user", transcript)
                            asyncio.create_task(enhanced_generate_and_speak(
                                user_text=transcript,
                                orchestrator=orchestrator,
                                call_state=call_state,
                                history=call_state.conversation_history,
                                speak_response_func=speak_response
                            ))
                        else:
                            call_state.empty_transcriptions += 1
                            asyncio.create_task(speak_recovery_phrase(call_state))
                    speech_buffer.clear()
        except asyncio.TimeoutError:
            if speech_buffer and (time.monotonic() - last_voice_ts) > SILENCE_GAP_S_LOCAL:
                duration_s = len(speech_buffer) / (8000 * 2)
                if duration_s >= MIN_SPEECH_S:
                    transcript = await asyncio.get_running_loop().run_in_executor(None, google_transcribe_pcm16_8k, bytes(speech_buffer))
                    call_state.total_transcriptions += 1
                    if transcript:
                        log.info(f"Transcrição final: '{transcript}'")
                        call_state.add_message("user", transcript)
                        asyncio.create_task(enhanced_generate_and_speak(
                            user_text=transcript,
                            orchestrator=orchestrator,
                            call_state=call_state,
                            history=call_state.conversation_history,
                            speak_response_func=speak_response
                        ))
                    else:
                        call_state.empty_transcriptions += 1
                        asyncio.create_task(speak_recovery_phrase(call_state))
                speech_buffer.clear()
            continue
        except Exception as e:
            log.error(f"Erro no loop STT: {e}", exc_info=True)
    log.info("STT processor encerrado.")

async def speak_recovery_phrase(call_state: CallState):
    if not call_state.opening_sequence_done or call_state.is_bot_speaking or call_state.is_generating_tts:
        return
    await asyncio.sleep(0.5)
    recovery_text = get_context_reengagement_message(call_state.context) if call_state.context else "Desculpe, não te ouvi bem. Pode repetir, por favor?"
    log.info(f"RECUPERAÇÃO: '{recovery_text}'")
    call_state.add_message("assistant", recovery_text)
    settings = voice_settings_for_affect("empathetic")
    await speak_response(call_state, recovery_text, settings, "live")
    call_state.awaiting_answer = True

async def monitor_user_silence(call_state: CallState, orchestrator: ImprovedOrchestrator):
    # Evita reengajar logo de cara se já havia histórico
    last_transcription_count = call_state.total_transcriptions

    # Cooldown entre reengajamentos (em segundos)
    REENGAGEMENT_COOLDOWN_S = 12.0
    if not hasattr(call_state, "last_reengagement_at"):
        call_state.last_reengagement_at = 0.0
    if not hasattr(call_state, "reengage_inflight"):
        call_state.reengage_inflight = False

    while call_state.ws_open:
        await asyncio.sleep(1.0)

        # Se o bot está falando ou ainda não esperamos resposta, não reengajar
        if call_state.is_bot_speaking or not call_state.awaiting_answer:
            continue

        # Se chegou transcrição nova do usuário, zera contador e segue
        if call_state.total_transcriptions > last_transcription_count:
            last_transcription_count = call_state.total_transcriptions
            continue

        # Precisamos ter um 'fim de fala' válido para medir silêncio
        if not getattr(call_state, "bot_speech_ended_at", 0):
            continue

        silence_duration = time.monotonic() - call_state.bot_speech_ended_at
        if silence_duration < REENGAGEMENT_TIMEOUT_S:
            continue

        # Respeitar cooldown entre reengajamentos
        if (time.monotonic() - call_state.last_reengagement_at) < REENGAGEMENT_COOLDOWN_S:
            continue

        # Evitar concorrência (um reengajamento por vez)
        if call_state.reengage_inflight:
            continue
        call_state.reengage_inflight = True
        try:
            # Mensagem contextual e dedupe contra a última fala do assistente
            msg = get_context_reengagement_message(getattr(orchestrator, "context", None)) or \
                  "Posso te ajudar com um seguro específico, como veículo, vida ou residência?"

            last_assistant = next(
                (m.get("text", "") for m in reversed(call_state.conversation_history) if m.get("role") == "assistant"),
                ""
            )
            if msg.strip() == last_assistant.strip():
                msg = "Só para confirmar: posso te ajudar com um seguro específico, como veículo, vida ou residência?"

            # Vamos falar: desligar 'awaiting_answer' até o fim do TTS (speak_response já liga is_bot_speaking)
            call_state.awaiting_answer = False
            await speak_response(call_state, msg, voice_settings_for_affect("empathetic"), "live")
            call_state.last_reengagement_at = time.monotonic()

            # Não altere last_transcription_count aqui; o cooldown + flags já previnem duplicação
        finally:
            call_state.reengage_inflight = False

def merge_short_fragments(call_state, new_text: str) -> Optional[str]:
    now = time.monotonic() * 1000
    t = (new_text or "").strip()
    if not t:
        return None
    if (call_state.pending_user_text and
        now - call_state.pending_since <= MERGE_WINDOW_MS and
        (len(t) <= SHORT_FRAGMENT_CHARS or len(call_state.pending_user_text) <= SHORT_FRAGMENT_CHARS)):
        combined = (call_state.pending_user_text + " " + t).strip()
        call_state.pending_user_text = None
        call_state.pending_since = 0
        return combined
    if len(t) <= SHORT_FRAGMENT_CHARS:
        call_state.pending_user_text = t
        call_state.pending_since = now
        return None
    return t

async def keep_alive(call_state: CallState):
    while call_state.ws_open:
        await asyncio.sleep(15)
        try:
            if call_state.ws_open and call_state.ws.client_state == WebSocketState.CONNECTED:
                await call_state.ws.send_json({"event": "ping"})
        except (WebSocketDisconnect, RuntimeError):
            break

def classify_conversation(history: List[dict], context: Optional[ConversationContext]) -> Dict[str, Any]:
    if not context:
        return {"label": "FALHA_CONTEXTO_AUSENTE", "confidence": 0.9, "reasons": ["O objeto de contexto da chamada não foi encontrado."]}
    user_msgs = [m for m in history if m.get("role") == "user"]
    if history and context.email:
        return {"label": "SUCESSO_EMAIL_CAPTURADO", "confidence": 0.95, "reasons": ["E-mail foi validado e armazenado."]}
    if context.reschedule_time:
         return {"label": "SUCESSO_REAGENDAMENTO", "confidence": 0.90, "reasons": ["Usuário pediu para ligar mais tarde e horário foi capturado."]}
    if len(user_msgs) <= 1 and not context.user_name:
        return {"label": "FALHA_ABANDONO_INICIAL", "confidence": 0.8, "reasons": ["Chamada encerrada antes da coleta de informações."]}
    if any("não quero" in m.get("text","").lower() for m in user_msgs):
        return {"label": "FALHA_RECUSA_EXPLICITA", "confidence": 0.85, "reasons": ["Usuário recusou explicitamente a oferta."]}
    return {"label": "INCONCLUSIVO", "confidence": 0.5, "reasons": ["A chamada foi encerrada sem um resultado claro."]}

def save_transcript_to_txt(call_state: CallState):
    if not call_state.conversation_history:
        log.info("Nenhum histórico de conversa para salvar.")
        return
    try:
        transcript_dir = "transcripts"
        os.makedirs(transcript_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_id = call_state.call_id.replace(":", "_")
        filepath = os.path.join(transcript_dir, f"transcript_{timestamp}_{filename_id}.txt")
        classification = classify_conversation(call_state.conversation_history, call_state.context)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("### --- DADOS COLETADOS ---\n")
            f.write(f"Nome: {call_state.context.user_name or 'Não informado'}\n")
            f.write(f"Email: {call_state.context.email or 'Não informado'}\n")
            f.write(f"Produto de Interesse: {call_state.context.insurance_type or 'Não informado'}\n")
            f.write(f"Horário para Retorno: {call_state.context.reschedule_time or 'Não informado'}\n")
            f.write("\n==============================\n")
            f.write(f"Call ID: {call_state.call_id}\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n")
            f.write(f"Resultado: {classification['label']} (Confiança: {classification['confidence']})\n")
            f.write(f"Motivos: {'; '.join(classification['reasons'])}\n")
            f.write("--- TRANSCRIÇÃO DA CHAMADA ---\n")
            seen_messages = set()
            for msg in call_state.conversation_history:
                role, text = msg.get("role"), msg.get("text")
                if not text or (text, role) in seen_messages:
                    continue
                seen_messages.add((text, role))
                f.write(f"[{role.upper()}]: {text}\n")
        log.info(f"Transcrição salva em: {filepath}")
    except Exception as e:
        log.error(f"Falha ao salvar a transcrição: {e}", exc_info=True)

# --- FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_queue, client_llm, client_el, client_openai
    tts_queue = Queue()
    client_llm = httpx.AsyncClient(
        base_url=LLM_BASE_URL,
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        timeout=30.0, http2=True
    )
    client_el = httpx.AsyncClient(
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Accept": "audio/mpeg"},
        timeout=30.0, http2=True
    )
    if OPENAI_API_KEY:
        client_openai = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=30.0, http2=True
        )
    asyncio.create_task(_tts_processor())
    yield
    if tts_queue:
        await tts_queue.put(None)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return PlainTextResponse("Livia Agent - OK")

@app.websocket("/test-ws-stream")
async def telnyx_media_ws(ws: WebSocket):
    await ws.accept()
    call_id = f"v5:{int(time.time() * 1000)}"
    call_state = CallState(call_id=call_id, ws=ws)
    log.info(f"Nova conexão WebSocket aceita. Call ID: {call_id}")
    try:
        agent = LiviaTurbinadaAgent(client_llm, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, fallback_client=client_openai)
        orchestrator = ImprovedOrchestrator(agent, rules_handler_func=rules_handler_func)
        call_state.context = orchestrator.context
    except Exception as e:
        log.error(f"Falha crítica ao inicializar a IA para {call_id}: {e}", exc_info=True)
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close(code=1008)
        return

    def get_dynamic_welcome_message() -> str:
        current_hour = datetime.now().hour
        greeting = "Boa tarde"
        if 5 <= current_hour < 12: greeting = "Bom dia"
        elif current_hour >= 18: greeting = "Boa noite"
        return f"{greeting}, aqui é a Livia, da Cardinalle Seguros."

    async def run_opening_sequence(ws: WebSocket, call_state: CallState):
        if call_state.opening_sequence_done: return
        log.info("Iniciando sequência de abertura com pausas.")
        await asyncio.sleep(1.5)
        greeting_text = get_dynamic_welcome_message()
        log.info(f"ABERTURA: '{greeting_text}'")
        call_state.add_message("assistant", greeting_text)        
        await speak_response(call_state, greeting_text, voice_settings_for_affect("friendly"), "live")
        if call_state.context:
            call_state.context.stage = "WELCOME"
            call_state.context.asked_questions.append(greeting_text)
        call_state.opening_sequence_done = True        
        log.info("Sequência de abertura concluída. Aguardando resposta do usuário.")

    audio_q = Queue()
    stt_task = asyncio.create_task(stt_processor_google(audio_q, call_state, orchestrator))
    keepalive_task = asyncio.create_task(keep_alive(call_state))
    monitor_task = asyncio.create_task(monitor_user_silence(call_state, orchestrator))
    opening_task: Optional[asyncio.Task] = None

    try:
        while call_state.ws_open:
            msg = await ws.receive_json()
            event = msg.get("event")
            if event == "start":
                start_data = msg.get("start", {})
                inbound_codec = (start_data.get("media_format", {}).get("encoding") or STREAM_CODEC).upper()
                call_state.codec_in = inbound_codec
                call_state.codec_out = inbound_codec
                log.info(f"Evento 'start' recebido. Codec unificado: {call_state.codec_out}")
                if opening_task is None or opening_task.done():
                    opening_task = asyncio.create_task(run_opening_sequence(ws, call_state))
            elif event == "media":
                track = msg.get("media", {}).get("track")
                if track != "inbound":
                    continue
                payload_b64 = msg.get("media", {}).get("payload")
                if not payload_b64:
                    continue
                raw_audio = base64.b64decode(payload_b64)
                if call_state.is_bot_speaking and not call_state.tts_cancel.is_set():
                    pcm16_audio = audioop.ulaw2lin(raw_audio, 2) if call_state.codec_in == "PCMU" else audioop.alaw2lin(raw_audio, 2)
                    rms = audioop.rms(pcm16_audio, 2)
                    is_after_grace_period = (time.monotonic() - (call_state.bot_phrase_started_at or 0)) > 0.3
                    if is_after_grace_period and rms > (SILENCE_RMS * 1.8):
                        call_state.voice_ms_during_tts += FRAME_MS
                    else:
                        call_state.voice_ms_during_tts = 0.0
                    if call_state.voice_ms_during_tts >= BARGE_MS and not call_state.barge_refractory():
                        log.info(f"Barge-in detectado para {call_id}. Cancelando fala da IA.")
                        call_state.tts_cancel.set()
                        call_state.last_barge_cancel_ts = time.monotonic()
                pcm16 = audioop.ulaw2lin(raw_audio, 2) if call_state.codec_in == "PCMU" else audioop.alaw2lin(raw_audio, 2)
                await audio_q.put(pcm16)
            elif event == "stop":
                log.info(f"Evento 'stop' recebido para {call_id}. Encerrando conexão.")
                call_state.ws_open = False
                await audio_q.put(None)
                break
    except WebSocketDisconnect:
        log.warning("WebSocket desconectado abruptamente.")
    except Exception as e:
        log.error(f"Erro no WebSocket: {e}", exc_info=True)
    finally:
        log.info(f"Iniciando limpeza de recursos para {call_id}.")
        call_state.ws_open = False
        if call_state.call_id and call_state.conversation_history:
            save_transcript_to_txt(call_state)
        tasks = [stt_task, keepalive_task, opening_task, monitor_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        try:
            if ws.client_state != WebSocketState.DISCONNECTED:
                await ws.close(code=1000)
        except RuntimeError as e:
            if "Unexpected ASGI message" in str(e):
                log.warning("Tentativa de fechar WebSocket já fechado. Ignorando.")
            else:
                raise e

# --- Telnyx API ---
class StartCallRequest(BaseModel):
    to: str

@app.post("/start_call")
async def start_call(req: StartCallRequest):
    to = req.to
    if not all([TELNYX_API_KEY, TELNYX_CONNECTION_ID, TELNYX_PHONE_NUMBER, to]):
        return JSONResponse({"error": "Configuração Telnyx incompleta."}, status_code=400)
    payload = {
        "connection_id": TELNYX_CONNECTION_ID,
        "to": to,
        "from": TELNYX_PHONE_NUMBER,
        "webhook_url": PUBLIC_HTTP_URL + "/test-webhook",
        "audio": {"codec": STREAM_CODEC},
    }
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    log.debug("Start call payload: %s", payload)
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as c:
        r = await c.post(f"{TELNYX_BASE_URL}/calls", json=payload)
        try:
            r.raise_for_status()
        except Exception as e:
            log.error("Falha ao iniciar chamada: %s | resp=%s", e, getattr(r, "text", ""))
            return JSONResponse({"error": "Telnyx falhou ao iniciar chamada."}, status_code=502)
        data = r.json()
        call_id = data.get("data", {}).get("id") or data.get("data", {}).get("call_control_id") or data.get("data", {}).get("call_control_id")
        call_control_id = data.get("data", {}).get("call_control_id", "")
        call_id = data.get("data", {}).get("id", call_control_id)
        call_control_id = data.get("data", {}).get("call_control_id", call_id)
    log.info("Chamada iniciada: call_id=%s, to=%s", call_control_id or call_id, to)
    return {"call_id": call_control_id or call_id, "status": "initiated"}

@app.post("/test-webhook")
async def test_webhook(request: Request):
    try:
        data = await request.json()
        evt = data.get("data", {})
        event = evt.get("event_type")
        payload = evt.get("payload", {})
        call_control_id = payload.get("call_control_id")
        log.info("Webhook recebido: event=%s call_id=%s", event, call_control_id)
        if event == "call.answered":
            log.info(f"Chamada atendida via webhook: call_id={call_control_id}")
            try:
                codec = "PCMA"
                json_body = {
                    "stream_url": PUBLIC_WSS_URL,
                    "stream_track": "both_tracks",
                    "stream_bidirectional_mode": "rtp",
                    "stream_bidirectional_codec": codec,
                    "stream_bidirectional_sampling_rate": 8000
                }
                async with httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {TELNYX_API_KEY}"}) as c:
                    resp = await c.post(f"{TELNYX_BASE_URL}/calls/{call_control_id}/actions/streaming_start", json=json_body)
                    resp.raise_for_status()
                log.info(f"streaming_start (both_tracks) → {resp.status_code} OK")
            except Exception as e:
                log.error("Falha CRÍTICA no streaming_start: %s", e)
            return JSONResponse({"status": "received"})
        elif event == "call.hangup":
            log.info("Chamada terminada via webhook: call_id=%s", call_control_id)
            return JSONResponse({"status": "received"})
        return JSONResponse({"status": "ignored", "event": event})
    except Exception as e:
        log.error("Erro em /test-webhook: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)