# improved_orchestration.py
import asyncio
import logging
import time
import json
import re
from enum import Enum
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import defaultdict
from datetime import datetime
from human_nuance import render_natural, affect_from_text, voice_settings_for_affect
from regras import get_default_question_for_stage

log = logging.getLogger(__name__)

class ConversationMode(Enum):
    SCRIPT_DRIVEN = "script"
    AI_ASSISTED = "ai_assisted"
    AI_DOMINANT = "ai"
    HYBRID = "hybrid"
    
def safe_asdict(obj):
    if is_dataclass(obj):
        return {f.name: safe_asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, defaultdict):
        return dict(obj)
    if isinstance(obj, dict):
        return {k: safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_asdict(v) for v in obj]
    return obj


@dataclass
class ConversationContext:
    mode: ConversationMode = ConversationMode.SCRIPT_DRIVEN
    off_script_count: int = 0
    fallback_count: int = 0
    last_ai_response: str = ""
    user_engagement_level: str = "neutral"
    conversation_quality: float = 1.0
    needs_recovery: bool = False
    intent_history: List[str] = field(default_factory=list)
    user_name: Optional[str] = None
    email_buffer: str = ""
    insurance_type: Optional[str] = None
    email: Optional[str] = None
    reschedule_time: Optional[str] = None
    asked_questions: List[str] = field(default_factory=list)
    last_question: Optional[str] = None
    user_input_history: List[Tuple[str, float]] = field(default_factory=list)
    question_blacklist: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stage: str = "OPENING"
    output_json: Dict[str, Any] = field(default_factory=lambda: {
        "name": None,
        "email": None,
        "reschedule_time": None,
        "product": None,
        "intent": None,
        "confidence": 0.0
    })
    name_attempts: int = 0
    email_attempts: int = 0

    def update_from_nlu(self, nlu_result: Dict[str, Any]):
        if nlu_result.get("name"):
            self.user_name = nlu_result["name"]
        if nlu_result.get("contact_time"):
            self.reschedule_time = nlu_result["contact_time"]
            self.stage = "END"
        if nlu_result.get("intent"):
            self.add_intent(nlu_result["intent"])
        if nlu_result.get("product"):
            self.insurance_type = nlu_result["product"]
        if nlu_result.get("email"):
            self.email = nlu_result["email"]

    def update_output_json(self, nlu_result: Dict[str, Any]):
        for key in ["name", "email", "reschedule_time", "product", "intent", "confidence"]:
            if key in nlu_result and nlu_result[key]:
                self.output_json[key] = nlu_result[key]

    def add_intent(self, intent: str):
        self.intent_history.append(intent)
        if len(self.intent_history) > 10:
            self.intent_history.pop(0)

    def add_user_input(self, text: str):
        self.user_input_history.append((text, time.monotonic()))
        if len(self.user_input_history) > 10:
            self.user_input_history.pop(0)

    def is_duplicate_input(self, text: str, window_sec: float = 5.0) -> bool:
        if not self.user_input_history:
            return False
        words = set(text.strip().lower().split())
        for prev_text, ts in self.user_input_history[-3:]:
            prev_words = set(prev_text.strip().lower().split())
            similarity = len(words & prev_words) / max(1, len(words | prev_words))
            if similarity > 0.8 and (time.monotonic() - ts) <= window_sec:
                return True
        return False

    def should_skip_question(self, question: str) -> bool:
        key = question.lower().strip()
        self.question_blacklist[key] += 1
        if "nome" in key and self.user_name:
            if self.question_blacklist[key] > 1:
                log.info(f"Bloqueando pergunta repetida: {question}")
                return True
        if self.question_blacklist[key] > 2:
            log.info(f"Bloqueando pergunta repetida (global): {question}")
            return True
        return False

class ImprovedOrchestrator:
    def __init__(self, livia_agent, rules_handler_func=None):
        self.agent = livia_agent
        self.context = ConversationContext()
        self.rules_handler_func = rules_handler_func

    def determine_conversation_mode(context):
        try:
            if context.get("confidence", 1.0) < 0.5 or "não" in user_text.lower():
                return "AI_DOMINANT"
            elif context.get("user_asked_question") or context.get("confidence", 1.0) < 0.65:
                mode = "AI_DOMINANT"
            elif context.get("stage") in ("RESCHEDULE", "COLLECT_DETAILS"):
                mode = "SCRIPT_DRIVEN"
            else:
                mode = "HYBRID"
        except Exception as e:
            log.exception("Erro ao determinar modo, fallback para HYBRID: %s", e)
            mode = "HYBRID"
        try:
            log.info("determine_conversation_mode -> %s", str(mode))
        except Exception:
            log.info("determine_conversation_mode -> (unprintable mode)")
        return mode

    def analyze_conversation_quality(self, user_text: str, history: List[dict], nlu_result: Dict[str, Any]) -> float:
        word_count = len(user_text.split())
        intent_confidence = nlu_result.get("confidence", 0.0)
        response_time = time.monotonic() - self.context.user_input_history[-1][1] if self.context.user_input_history else 1.0
        engagement_score = min(word_count / 5.0, 1.0) * 0.4 + intent_confidence * 0.4 + min(1.0 / response_time, 1.0) * 0.2
        self.context.conversation_quality = max(0.1, min(1.0, engagement_score))
        return self.context.conversation_quality

    def update_engagement_metrics(self, user_text: str, response_time: float):
        word_count = len(user_text.split())
        if word_count > 5:
            self.context.user_engagement_level = "high"
        elif word_count > 2:
            self.context.user_engagement_level = "medium"
        else:
            self.context.user_engagement_level = "low"
        if self.context.fallback_count > 2:
            self.context.conversation_quality = max(0.1, min(1.0, self.context.conversation_quality - 0.1))

    def get_context_summary(self) -> str:
        return (f"Stage={self.context.stage}, Mode={self.context.mode.value}, "
                f"UserName={self.context.user_name}, InsuranceType={self.context.insurance_type}, "
                f"Email={self.context.email}, Engagement={self.context.user_engagement_level}, "
                f"Quality={self.context.conversation_quality:.2f}")

    async def process_user_input(self, user_text: str, nlu_result: dict, history: List[dict], stage: str, last_question: str, call_state: Any):
        self.context.update_from_nlu(nlu_result)
        self.context.update_output_json(nlu_result)
        if callable(self.rules_handler_func):
            response, is_question, needs_ai_fallback, next_stage = await self.rules_handler_func(
                self, user_text, nlu_result, history, stage, last_question, call_state=call_state
            )           
        else:
            response = ""
            is_question = False
            needs_ai_fallback = True
            next_stage = stage
        if next_stage:
            self.context.stage = next_stage
        log.info(f"Rules handle retornou: response='{response}', next_stage='{self.context.stage}'")
        self.context.add_user_input(user_text)
        return response, is_question, needs_ai_fallback, self.context.stage

# --- Custom Exception Types for Granular Error Handling ---
class NLUProcessingError(Exception):
    pass

class ResponseGenerationError(Exception):
    pass

class TTSPlaybackError(Exception):
    pass

# --- Constants ---
PROCESS_TIMEOUT_SECONDS = 15
LOCK_ACQUIRE_TIMEOUT_SECONDS = 0.1

async def enhanced_generate_and_speak(
    user_text: str,
    orchestrator: Any,
    call_state: Any,
    history: List[dict],
    speak_response_func: Callable
):
    if call_state.conversation_ended:
        log.warning("Conversation already ended. Ignoring new user input.")
        return
    if not hasattr(call_state, 'processing_lock'):
        call_state.processing_lock = asyncio.Lock()
    try:
        await asyncio.wait_for(call_state.processing_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        log.error("Failed to acquire processing lock, another process may be stuck.")
        return
    except Exception as e:
        log.error(f"Unexpected error acquiring lock: {e}", exc_info=True)
        return
    start_time = time.monotonic()
    try:
        main_task = asyncio.create_task(_process_and_speak_task(
            user_text, orchestrator, call_state, history, speak_response_func
        ))
        await asyncio.wait_for(main_task, timeout=PROCESS_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        log.warning("Main processing task was cancelled.")
        raise
    except asyncio.TimeoutError:
        log.error(f"Overall processing timed out after {PROCESS_TIMEOUT_SECONDS}s.")
        await _speak_error_response(call_state, speak_response_func, "Desculpe, estou com um problema técnico. Pode repetir, por favor?")
    except (NLUProcessingError, ResponseGenerationError) as e:
        log.error(f"Recoverable application error: {e}", exc_info=True)
        await _speak_error_response(call_state, speak_response_func, "Desculpe, não entendi muito bem. Pode dizer de novo?")
    except TTSPlaybackError as e:
        log.error(f"Failed to speak response: {e}", exc_info=True)
    except Exception as e:
        log.critical(f"CRITICAL unhandled error in orchestrated processing: {e}", exc_info=True)
        await _speak_error_response(call_state, speak_response_func, "Encontrei um erro no sistema. Por favor, tente ligar mais tarde.")
    finally:
        if call_state.processing_lock.locked():
            call_state.processing_lock.release()

async def _process_and_speak_task(user_text, orchestrator, call_state, history, speak_response_func):
    start_time = time.monotonic()
    await _handle_cancellation(call_state)
    nlu_result = await _perform_nlu(orchestrator, user_text)
    orchestrator.context.update_from_nlu(nlu_result)
    orchestrator.context.update_output_json(nlu_result)
    response_text, is_question = await _determine_response(
        orchestrator, user_text, nlu_result, history, call_state
    )
    await _speak_and_update_state(
        call_state, orchestrator, response_text, is_question, user_text, speak_response_func
    )
    orchestrator.update_engagement_metrics(user_text, time.monotonic() - start_time)
    log.info(f"CONTEXT: {orchestrator.get_context_summary()}")

async def _handle_cancellation(call_state: Any):
    if hasattr(call_state, "tts_cancel") and call_state.tts_cancel is not None:
        log.info("Cancelling previous TTS task.")
        call_state.tts_cancel.set()
        await asyncio.sleep(0.05)
    call_state.tts_cancel = asyncio.Event()

async def _perform_nlu(orchestrator: Any, user_text: str) -> Dict:
    user_text = (user_text or "").strip()
    if not user_text:
        return {}
    try:
        current_stage = orchestrator.context.stage
        asked = orchestrator.context.asked_questions
        last_question = asked[-1] if asked else "starting conversation"
        log.info(f"DIAGNOSTIC: Processing ---> '{user_text}' (Stage: {current_stage})")
        nlu_result = await orchestrator.agent.nlu_router(
            user_input=user_text, stage=current_stage, last_question=last_question
        )
        log.info(f"DIAGNOSTIC: NLU ---> {nlu_result}")
        return nlu_result
    except Exception as e:       
        raise NLUProcessingError(f"NLU router failed: {e}") from e

def summary_to_dict(summary: str) -> Dict[str, Any]:
    result = {}
    items = summary.split(", ")
    for item in items:
        parts = item.split("=", 1)
        if len(parts) == 2:
            key, value = parts
            result[key] = value if value and value != 'None' else None
    return result

async def _determine_response(   
    orchestrator: Any, 
    user_text: str, 
    nlu_result: Dict, 
    history: List[dict], 
    call_state: Any
) -> Tuple[str, bool]:
    try:
        last_question = orchestrator.context.asked_questions[-1] if orchestrator.context.asked_questions else ""
        result = await orchestrator.process_user_input(
            user_text, nlu_result, history, orchestrator.context.stage, last_question, call_state=call_state
        )
        # Normaliza diferentes formatos de retorno para segurança
        if isinstance(result, tuple):
            if len(result) == 4:
                response, is_question, needs_ai_fallback, next_stage = result
            elif len(result) == 3:
                response, is_question, needs_ai_fallback = result
                next_stage = orchestrator.context.stage
            elif len(result) == 2:
                response, next_stage = result
                is_question = response.strip().endswith('?')
                needs_ai_fallback = False
            elif len(result) == 1:
                response = result[0]
                is_question = response.strip().endswith('?')
                needs_ai_fallback = False
                next_stage = orchestrator.context.stage
            else:
                response = ""
                is_question = False
                needs_ai_fallback = True
                next_stage = orchestrator.context.stage
        else:
            response = str(result) if result is not None else ""
            is_question = response.strip().endswith('?')
            needs_ai_fallback = False
            next_stage = orchestrator.context.stage

        # Mantém o estágio coerente
        orchestrator.context.stage = next_stage

        if next_stage == "END" and hasattr(call_state, "conversation_ended"):
            call_state.conversation_ended = True
            # --- Recuperação inteligente para e-mail fragmentado / confirmação ---
        stage_before = orchestrator.context.stage
            
        # 1) Se estamos pedindo e-mail, tente extrair de fala ambígua (arroba/ponto/underscore)
        if stage_before == "ASK_EMAIL" and not response:
            email_guess = _maybe_email_from_text(user_text)
            if email_guess:
                orchestrator.context.email = email_guess
                response = f"Anotei {email_guess.replace('@', ' arroba ').replace('.', ' ponto ')} , está correto?"
                next_stage = "CONFIRM_EMAIL"
                orchestrator.context.stage = next_stage

        # 2) Se estamos confirmando e-mail e o usuário respondeu algo do tipo "sim", "tá certo", "correto", avance
        if orchestrator.context.stage == "CONFIRM_EMAIL" and not response:
            low = user_text.strip().lower()
            affirm_set = {"sim", "certo", "ok", "claro", "tá certo", "ta certo", "perfeito", "isso", "correto", "agora sim", "está correto", "esta correto"}
            if any(a in low for a in affirm_set):
                # confirmação tácita
                response = "Obrigado. Qual o melhor dia e horário para nosso especialista te ligar?"
                next_stage = "ASK_WHEN_TO_CALL"
                orchestrator.context.stage = next_stage

        # Fallback de IA se necessário
        if needs_ai_fallback:
            from dataclasses import asdict, is_dataclass
            
            def _context_to_dict(ctx):
                # Tenta dataclass.asdict
                try:
                    if is_dataclass(ctx):
                        return asdict(ctx)
                except Exception:
                    pass
                # Tenta vars()
                try:
                    return {k: v for k, v in vars(ctx).items() if not callable(v)}
                except Exception:
                    # Tenta dir() como último recurso
                    out = {}
                    for k in dir(ctx):
                        if k.startswith("_"):
                            continue
                        try:
                            v = getattr(ctx, k)
                        except Exception:
                            continue
                        if not callable(v):
                            out[k] = v
                    return out

            context_copy = _context_to_dict(orchestrator.context)
            ai_answer = await orchestrator.agent.process_user_response(user_text, history, context_copy, nlu_result)

            # --- LÓGICA DE RECUPERAÇÃO APRIMORADA ---
            # Palavras-chave que indicam uma resposta fraca/genérica da IA
            WEAK_AI_RESPONSE_KEYWORDS = ["desculpe", "lamento", "não consegui", "não sei", "infelizmente"]
            
            # Verifica se a resposta da IA é inútil
            is_weak_response = (
                not ai_answer.strip() or
                any(keyword in ai_answer.strip().lower() for keyword in WEAK_AI_RESPONSE_KEYWORDS)
            )

            if is_weak_response:
                log.warning(f"Resposta fraca da IA recebida: '{ai_answer}'. Retomando com pergunta padrão do estágio '{orchestrator.context.stage}'.")
                # Chama a função de recuperação com o objeto de contexto completo
                response = get_default_question_for_stage(orchestrator.context)
            else:
                response = ai_answer
            # --- FIM DA LÓGICA DE RECUPERAÇÃO ---
          
        if not response:
            response = get_default_question_for_stage(orchestrator.context)

        return response, is_question
    except Exception as e:
        raise ResponseGenerationError(f"Response logic failed: {e}") from e
        
async def _speak_and_update_state(
    call_state: Any,
    orchestrator: Any,
    response_text: str,
    is_question: bool,
    user_text: str,
    speak_response_func: Callable,
) -> None:
    try:
        affect = "professional"
        final_response = response_text or ""

        # normaliza pontuação de pergunta
        if is_question and not final_response.strip().endswith("?"):
            final_response = final_response.rstrip(".…!") + "?"

        # de-dupe de perguntas repetidas (patch)
        if final_response.strip().endswith("?"):
            # import local para evitar import circular
            from regras import get_default_question_for_stage
            if orchestrator.context.should_skip_question(final_response):
                alt = get_default_question_for_stage(orchestrator.context)
                if alt.strip().lower() == final_response.strip().lower():
                    final_response = "Entendido."
                    is_question = False
                else:
                    final_response = alt
                    is_question = final_response.strip().endswith("?")

        if is_question:
            orchestrator.context.asked_questions.append(final_response)

        settings = {}
        log.info(f"FINAL RESPONSE: '{final_response}'")
        call_state.add_message("assistant", final_response)
        
        call_state.awaiting_answer = is_question
        await speak_response_func(call_state, final_response, settings, "live")

    except Exception as e:
        log.error(f"Failed inside _speak_and_update_state: {e}", exc_info=True)
        # fala de recuperação para não ficar mudo
        await _speak_error_response(
            call_state,
            speak_response_func,
            "Desculpe, tive um problema aqui. Podemos continuar?"
        )


async def _speak_error_response(call_state: Any, speak_response_func: Callable, message: str) -> None:
    try:
        log.info(f"Speaking recovery response: '{message}'")
        await speak_response_func(call_state, message, {}, "live")
    except Exception as recovery_e:
        log.error(f"Failed to even send the recovery response: {recovery_e}", exc_info=True)
_EMAIL_VALID_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

def _normalize_email_tokens(text: str) -> str:
    """
    Converte fala 'natural' em e-mail:
    - 'arroba' -> '@' ; 'ponto' -> '.'
    - remove espaços e underscores
    - tenta completar domínios comuns se o usuário falar 'hotmail', 'gmail', etc.
    """
    s = text.strip().lower()

    s = s.replace(" arroba ", "@").replace(" arroba", "@").replace("arroba ", "@").replace("arroba", "@")
    s = s.replace(" ponto ", ".").replace(" ponto", ".").replace("ponto ", ".").replace("ponto", ".")

    s = s.replace(" ", "").replace("_", "")

    if "@" in s:
        local, _, dom = s.partition("@")
        if dom and "." not in dom:
            # tenta completar domínios mais comuns
            mapping = {
                "gmail": "gmail.com",
                "hotmail": "hotmail.com",
                "outlook": "outlook.com",
                "yahoo": "yahoo.com.br",
                "uol": "uol.com.br",
                "bol": "bol.com.br",
                "terra": "terra.com.br",
                "live": "live.com",
                "icloud": "icloud.com",
            }
            dom = mapping.get(dom, dom)
            s = f"{local}@{dom}"

    return s

def _maybe_email_from_text(text: str) -> str:
    cand = _normalize_email_tokens(text)
    return cand if _EMAIL_VALID_RE.match(cand or "") else ""
        