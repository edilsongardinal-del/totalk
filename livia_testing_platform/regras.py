# regras.py
import re
import random
import logging
from collections import defaultdict
import time
from typing import Tuple, List, Dict, Any, Optional
from email_validator import validate_email, EmailNotValidError
log = logging.getLogger(__name__)

# Configurações e constantes
NAME_CONFIRM_RE = re.compile(r"\b(qual é|qual eh|como é|como eh|quem é|quem eh|nome da agente)\b", re.IGNORECASE)
COMPANY_CONFIRM_RE = re.compile(r"\b(qual é|qual eh|nome da empresa|quem são|quem sao)\b", re.IGNORECASE)
OBJECTION_CALL_LATER = re.compile(r"\b(ligar depois|mais tarde|outro dia|agora não|agora nao|ocupado)\b", re.IGNORECASE)
OBJECTION_HAVE_INSURANCE = re.compile(r"\b(já tenho|ja tenho|já possuo|ja possuo)\b", re.IGNORECASE)
AFFIRM = {"sim", "claro", "pode", "ok", "certo", "com certeza"}
TIME_HINTS = ["hoje", "amanhã", "amanha", "segunda", "terça", "terca", "quarta", "quinta", "sexta", "sábado", "sabado", "domingo"]
NAME_REPROMPTS = [
    "Desculpe, não peguei seu nome direitinho. Pode repetir, por favor?",
    "Não entendi bem seu nome. Pode dizer de novo?",
    "Qual é seu nome mesmo, por favor?"
]
INTEREST_REPROMPTS = [
    "Você gostaria de saber mais sobre seguro para veículo, vida, viagem, saúde ou residência?",
    "Algum seguro específico te interessa, como auto, vida ou casa?",
    "Que tipo de seguro você está pensando? Auto, vida, saúde, viagem ou residência?"
]
EMAIL_REPROMPTS_FRESH = [
    "Desculpe, não consegui entender o e-mail. Pode dizer de novo, por favor?",
    "Não peguei o e-mail certinho. Pode repetir, por gentileza?",
    "Qual é o seu e-mail mesmo? Pode dizer com calma."
]
_COMMON_DOMAINS = {
    "hotmail": "hotmail.com", "gmail": "gmail.com", "yahoo": "yahoo.com.br",
    "outlook": "outlook.com", "uol": "uol.com.br", "bol": "bol.com.br",
    "terra": "terra.com.br", "live": "live.com"
}
STOP_WORDS_AS_NAME = {
    "sobre", "quero", "saude", "vida", "viagem", "residencia", "casa", 
    "carro", "auto", "seguro", "sim", "não", "nao", "ok", "claro", "pode",
    "podemos", "falar", "comigo", "meu", "nome", "é", "valor", "qual",
    "boa", "tarde", "dia", "noite", "oi", "ola", "olá"
}
GREETING_RE = re.compile(r"\b(boa\s*(tarde|noite)|bom\s*dia|oi|ol[aá])\b", re.I)
BYE_RE = re.compile(r"\b(valeu|tchau|até mais|ate mais)\b", re.I)

def display_name(context) -> str:
    return context.user_name or "você"

def extract_day_hint(text: str) -> str:
    text = text.lower()
    for day in TIME_HINTS:
        if day in text:
            return day
    return ""

def normalize_contact_time_phrase(text: str) -> str:
    return text.strip().lower()

def is_likely_name(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    if len(t.split()) == 1 and t in STOP_WORDS_AS_NAME:
        return False
    parts = t.split()
    if any(part in STOP_WORDS_AS_NAME for part in parts):
        return False
    if len(parts) >= 1 and all(len(p) >= 2 for p in parts):
        return True
    return False

def _detect_product(text: str) -> str:
    text = (text or "").strip().lower()
    patterns = {
        "auto": r"\b(carro|autom[óo]vel|ve[íi]culo)\b",
        "vida": r"\b(vida|seguro de vida)\b",
        "residencial": r"\b(casa|im[óo]vel|res(id[êe]ncial?)?)\b",
        "viagem": r"\b(viagem|travel)\b",
        "saúde": r"\b(sa[úu]de|plano de sa[úu]de)\b",
        "previdência": r"\b(previd[êe]ncia)\b",
    }
    for product, pattern in patterns.items():
        if re.search(pattern, text):
            return product
    return ""

def make_reschedule_prompt(context, last_user_utterance: str) -> str:
    user_name = display_name(context)
    hint = extract_day_hint(last_user_utterance)
    period_q = "Prefere de manhã, depois do almoço ou à tarde?"
    if hint:
        base = f"Perfeito, {user_name}. {hint.capitalize()} te atende bem?"
        return f"{base} {period_q}"
    else:
        base = f"Perfeito, {user_name}. Qual dia te atende melhor?"
        return f"{base} {period_q}"

def _bridge_answer_then_resume(context, short_answer: str) -> str:
    last_question = context.asked_questions[-1] if context.asked_questions else "Para continuarmos, pode me dizer seu nome, por favor?"
    if "?" in last_question:
        parts = re.split(r'(?<=[.!?])\s*', last_question)
        question_part = [part for part in parts if '?' in part]
        if question_part:
            resume_q = question_part[-1]
        else:
            resume_q = last_question
    else:
        resume_q = last_question
    return f"{short_answer} {resume_q}"

def _normalize_spoken_email_piece(t: str) -> str:
    if not t: return ""
    s = t.strip().lower().replace(' ', '')
    s = s.replace(" ", "").replace("’", "'").replace("`", "'")
    s = s.replace("arroba", "@").replace("ponto", ".").replace("dot", ".")
    s = s.replace("hífen", "-").replace("hifen", "-")
    s = s.replace("underline", "_").replace("traço", "-")
    s = s.replace("gmailcom", "gmail.com").replace("gmail.con", "gmail.com")
    s = s.replace("yahoocombr", "yahoo.com.br")
    s = s.replace("uolcombr", "uol.com.br")
    s = s.replace("outlookcom", "outlook.com")
    s = s.strip(" ,;:!?/\\")
    s = re.sub(r'^(?:\w+\s+)*([\w.-]+@[\w.-]+.[\w]{2,})$', r'\1', s)
    return s

def extract_email_candidates(text: str) -> List[str]:
    text = _normalize_spoken_email_piece(text)
    matches = re.findall(r'([\w\.-]+(?:\s+[\w\.-]+)*@[\w\.-]+\.[\w]{2,})', text)
    return [m.strip().replace(' ', '') for m in matches]

def validate_and_clean_email(email: str) -> Tuple[bool, str]:
    if not email: return (False, "")  
    pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'

    cleaned = email.strip()

    if '@' in cleaned:
        user_part, domain_part = cleaned.split('@', 1)
        if domain_part and '.' not in domain_part and domain_part in _COMMON_DOMAINS:
            cleaned = f"{user_part}@{_COMMON_DOMAINS[domain_part]}"
    ok = bool(re.match(pattern, cleaned, re.IGNORECASE))
    return (ok, cleaned)

def assemble_email_fragmented(buffer_text: str, new_piece: str) -> str:
    if not new_piece:
        return buffer_text
    candidates = extract_email_candidates(new_piece)
    if candidates:
        is_valid, cleaned = validate_and_clean_email(candidates[0])
        if is_valid:
            return cleaned
    combined = (buffer_text + new_piece).replace(" ", "")
    return _normalize_spoken_email_piece(combined)

_question_counts = defaultdict(int)
_question_last_time = {}
QUESTION_REPEAT_LIMIT = 2
QUESTION_COOLDOWN_S = 60

def should_skip_question(q_text: str, context: dict) -> bool:
    if not q_text:
        return True
    q_key = q_text.strip().lower()
    _question_counts[q_key] += 0
    if _question_counts[q_key] >= QUESTION_REPEAT_LIMIT:
        last = _question_last_time.get(q_key, 0)
        if (time.time() - last) < QUESTION_COOLDOWN_S:
            return True
    return False
#====================================================================================
async def rules_handler_func(orchestrator, user_text, nlu, history, stage, expected):
    ctx = orchestrator.context
    intent = (nlu or {}).get("intent", "OTHER")
    response_text = ""
    needs_ai = True
    is_question = False
    next_stage = stage

    if intent == "END_CONVERSATION":
        response_text = "Tudo bem, agradeço seu tempo. Tenha um ótimo dia!"
        next_stage = "END"
        needs_ai = False
        is_question = False
        return response_text, is_question, needs_ai, next_stage


    if intent == "RESCHEDULE_REQUEST":
        response_text = "Sem problemas."
        next_stage = "ASK_WHEN_TO_CALL"
        needs_ai = False

    elif intent == "GIVE_WHEN":
        when = (nlu or {}).get("when")
        if when:
            # limpa prefixos e pontuação final
            when_clean = re.sub(r'(?i)^(pode ser|poderia ser|seria|talvez)\s+', '', when).strip()
            when_clean = when_clean.strip(" .,!;")
            when_clean = when_clean.replace("manha", "manhã")
            ctx.reschedule_time = when_clean
            response_text = f"Perfeito, agendo para {when_clean}."
            next_stage = "END"
            needs_ai = False


    if intent == "GIVE_NAME":
        name = (nlu or {}).get("name")
        if name:
            ctx.user_name = name
            response_text = f"Perfeito, {name}."
            next_stage = "ASK_INTEREST"
            needs_ai = False

    elif intent == "CHOOSE_PRODUCT":
        prod = (nlu or {}).get("product")
        if prod:
            ctx.insurance_type = prod
            response_text = "Ótimo."
            next_stage = "ASK_EMAIL"
            needs_ai = False

    elif intent == "GIVE_EMAIL":
        email = (nlu or {}).get("email")
        if email:
            ctx.email = email
            response_text = "Obrigado."
            next_stage = "ASK_WHEN_TO_CALL"
            needs_ai = False

    elif intent in ("RESCHEDULE_REQUEST", "USER_RESISTANCE"):
        response_text = "Sem problemas."
        next_stage = "ASK_WHEN_TO_CALL"
        needs_ai = False

    # se já temos tudo e a próxima etapa era marcar horário
    if getattr(ctx, "email", None) and getattr(ctx, "insurance_type", None) and next_stage == "ASK_WHEN_TO_CALL":
        # o app acrescenta a pergunta via >>> RETOMAR
        pass

    return response_text, is_question, needs_ai, next_stage
    
def record_question_if_said(q_text: str, context: dict):
    if not q_text:
        return
    q_key = q_text.strip().lower()
    context.setdefault("asked_questions", [])
    if not context["asked_questions"] or context["asked_questions"][-1] != q_text:
        context["asked_questions"].append(q_text)
    _question_counts[q_key] += 1
    _question_last_time[q_key] = time.time()


def get_default_question_for_stage(ctx) -> str:
    stage = getattr(ctx, "stage", "ASK_NAME")
    if stage == "ASK_NAME":
        return "Para começar, qual é o seu nome, por favor?"
    if stage == "ASK_INTEREST":
        return "Você tem interesse em algum seguro específico? (auto, vida, residencial, viagem ou previdência)"
    if stage == "ASK_EMAIL":
        return "Qual é o seu melhor e-mail para enviarmos os próximos passos?"
    if stage == "ASK_WHEN_TO_CALL":
        return "Qual é o melhor dia/horário para retornarmos a ligação?"
    return ""
    
def get_current_question(context):
    if context.stage == "ASK_NAME" and not context.user_name:
        return "Qual é o seu nome, por favor?"
    elif context.stage == "ASK_HAS_INSURANCE" and context.user_name:
        return f"{context.user_name}, você já possui algum tipo de seguro atualmente?"
    elif context.stage == "ASK_INTEREST":
        return "Você teria interesse em algum seguro para veículo, vida, viagem, saúde ou residência?"
    elif context.stage == "ASK_EMAIL" and not context.email:
        return "Poderia me passar seu melhor e-mail, por favor?"
    elif context.stage == "ASK_WHEN_TO_CALL":
        return "Qual o melhor dia e horário para nosso especialista te ligar?"
    else:
        return "Como posso te ajudar?"

def get_context_reengagement_message(context):
    if getattr(context, "asked_questions", None):
        return context.asked_questions[-1]
    return get_current_question(context)

def make_email_speakable(email: str) -> str:
    return email.replace("@", " arroba ").replace(".", " ponto ")

def sanitize_speech(text: str) -> str:
    if not text: return text
    s = text.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith('“') and s.endswith('”')):
        s = s[1:-1].strip()
    s = re.sub(r'^[“"]?\s*não parece que você[^?!.:,]*[:.,-]?\s*','',s,flags=re.IGNORECASE)
    return s.strip()

def robust_extract_email(text: str) -> Optional[str]:
    if not text: return None
    t = text.lower().replace("arroba", "@").replace("ponto", ".")
    match = re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", t)
    return match.group(0) if match else None

def validate_email_with_retry(email: str, max_retries: int = 2) -> Tuple[bool, str]:
    for attempt in range(max_retries + 1):
        try:
            validated_email = validate_email(email, check_deliverability=False)
            return True, validated_email.email
        except EmailNotValidError as e:
            log.warning(f"Validação de e-mail falhou (tentativa {attempt + 1}): {email}, Erro: {e}")
            if attempt == max_retries:
                return False, str(e)
            time.sleep(0.5)
    return False, "Erro desconhecido após tentativas"

def extract_day_hint_enhanced(text: str) -> str:
    text = text.lower().strip()
    for hint in TIME_HINTS:
        if hint in text:
            return hint
    return ""

def normalize_contact_time_phrase_enhanced(text: str) -> str:
    return text.strip().lower()
    

