# -*- coding: utf-8 -*-
# improved_rules.py - Substitua ou melhore as regras existentes

import re
import random
import logging
from collections import defaultdict
import time
from typing import Tuple, List, Dict, Any, Optional
from email_validator import validate_email, EmailNotValidError

log = logging.getLogger(__name__)

# Contador global para detectar loops
_question_repetition_count = defaultdict(int)
_last_question_time = defaultdict(float)

# Respostas de recuperação quando há muito loop
RECOVERY_RESPONSES = {
    "email_loop": [
        "Vou facilitar: só preciso de um email válido para nosso especialista te contatar. Pode ser gmail, hotmail, qualquer um que você usa.",
        "Para não complicar: qual email você usa no dia a dia? Gmail, Hotmail, Yahoo?",
        "Simples assim: me dá seu email principal que nosso especialista vai te mandar uma proposta."
    ],
    "name_loop": [
        "Só preciso saber como te chamar. Qual seu primeiro nome?",
        "Para personalizar o atendimento: como você gostaria que eu te chamasse?",
        "Me diz só seu primeiro nome para continuarmos."
    ],
    "product_loop": [
        "Vou direto ao ponto: temos seguro para carro, casa, vida e viagem. Qual te interessa?",
        "Simples: você quer seguro para que? Carro, casa ou vida?",
        "Sem complicar: qual proteção você procura? Auto, residencial ou vida?"
    ],
    "schedule_loop": [
        "Para nosso especialista te ligar: prefere manhã, tarde ou noite?",
        "Horário para retorno: melhor de manhã ou à tarde para você?",
        "Quando é melhor te ligarmos de volta? Manhã, tarde ou noite?"
    ]
}

def detect_conversation_loop(context, current_stage: str, user_response: str) -> bool:
    """Detecta se a conversa entrou em loop infinito"""
    
    loop_indicators = [
        "não entendi", "pode ser mais direto", "como assim", "não captei",
        "pode repetir", "não compreendi", "explica melhor"
    ]
    
    # Se usuário repetiu confusão múltiplas vezes
    confusion_count = sum(1 for indicator in loop_indicators 
                         if indicator in user_response.lower())
    
    if confusion_count > 0:
        stage_key = f"{current_stage}_confusion"
        _question_repetition_count[stage_key] += 1
        
        # Loop detectado após 2 confusões no mesmo estágio
        return _question_repetition_count[stage_key] >= 2
    
    return False

def get_recovery_response(context, stage: str, loop_type: str = "general") -> str:
    """Gera resposta de recuperação para quebrar loops"""
    
    stage_key = f"{stage.lower()}_loop"
    
    if stage_key in RECOVERY_RESPONSES:
        return random.choice(RECOVERY_RESPONSES[stage_key])
    
    # Fallback genérico
    fallbacks = [
        "Vou tentar de outro jeito: o que posso esclarecer para você?",
        "Deixa eu reformular: em que posso te ajudar especificamente?",
        "Para não complicar: me diz o que não ficou claro."
    ]
    
    return random.choice(fallbacks)

def is_valid_email_attempt(text: str) -> bool:
    """Verifica se o texto parece uma tentativa válida de email"""
    
    # Normaliza texto falado para email
    normalized = text.lower()
    normalized = normalized.replace(" arroba ", "@").replace(" ponto ", ".")
    normalized = normalized.replace("arroba", "@").replace("ponto", ".")
    
    # Padrões que indicam tentativa de email
    email_patterns = [
        r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}',
        r'[a-zA-Z0-9]+\s*arroba\s*[a-zA-Z0-9]+',
        r'[a-zA-Z0-9]+@[a-zA-Z0-9]+',
        r'[a-zA-Z0-9]+\s*@\s*[a-zA-Z0-9]+'
    ]
    
    return any(re.search(pattern, normalized) for pattern in email_patterns)

def smart_email_extraction(text: str) -> Optional[str]:
    """Extração inteligente de email de texto falado"""
    
    # Normalização agressiva
    normalized = text.lower().strip()
    
    # Substitui palavras faladas por símbolos
    replacements = {
        " arroba ": "@", "arroba": "@",
        " ponto ": ".", "ponto": ".",
        " hifen ": "-", "hífen": "-",
        " underline ": "_", " underscore ": "_"
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # Remove espaços desnecessários
    normalized = re.sub(r'\s+', '', normalized)
    
    # Completa domínios comuns
    domain_completions = {
        "@gmail": "@gmail.com",
        "@hotmail": "@hotmail.com", 
        "@yahoo": "@yahoo.com.br",
        "@outlook": "@outlook.com",
        "@uol": "@uol.com.br"
    }
    
    for incomplete, complete in domain_completions.items():
        if incomplete in normalized and complete not in normalized:
            normalized = normalized.replace(incomplete, complete)
    
    # Tenta validar
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, normalized)
    
    if match:
        email = match.group(0)
        try:
            validate_email(email, check_deliverability=False)
            return email
        except EmailNotValidError:
            pass
    
    return None

async def improved_rules_handler_func(orchestrator, user_text, nlu, history, stage, expected):
    """Versão melhorada das regras com detecção de loops"""
    
    ctx = orchestrator.context
    intent = (nlu or {}).get("intent", "OTHER")
    response_text = ""
    needs_ai = True
    is_question = False
    next_stage = stage
    
    # Detecta loop infinito
    if detect_conversation_loop(ctx, stage, user_text):
        log.warning(f"Loop detectado no estágio {stage}. Aplicando recuperação.")
        
        response_text = get_recovery_response(ctx, stage)
        needs_ai = False
        is_question = True
        
        # Reset do contador após recuperação
        stage_key = f"{stage}_confusion"
        _question_repetition_count[stage_key] = 0
        
        return response_text, is_question, needs_ai, next_stage
    
    # Lógica normal das regras com melhorias
    
    if intent == "RESCHEDULE_REQUEST":
        response_text = "Sem problemas. Quando prefere que liguemos?"
        next_stage = "ASK_WHEN_TO_CALL"
        needs_ai = False
        is_question = True

    elif intent == "GIVE_WHEN":
        when = (nlu or {}).get("when")
        if when:
            when_clean = re.sub(r'(?i)^(pode ser|poderia ser|seria|talvez)\s+', '', when).strip()
            when_clean = when_clean.strip(" .,!;")
            ctx.reschedule_time = when_clean
            response_text = f"Perfeito, agendo para {when_clean}. Nosso especialista vai te ligar. Obrigada pelo seu tempo!"
            next_stage = "END"
            needs_ai = False

    elif intent == "GIVE_NAME":
        name = (nlu or {}).get("name")
        if name and name.lower() not in ["não", "nao", "nada"]:
            ctx.user_name = name
            response_text = f"Prazer, {name}!"
            next_stage = "ASK_INTEREST"
            needs_ai = False

    elif intent == "CHOOSE_PRODUCT":
        prod = (nlu or {}).get("product")
        if prod:
            ctx.insurance_type = prod
            response_text = f"Entendi, {prod}. Ótima escolha!"
            next_stage = "ASK_EMAIL"
            needs_ai = False

    elif intent == "GIVE_EMAIL":
        email = (nlu or {}).get("email")
        if email:
            ctx.email = email
            response_text = f"Anotei: {email}. Perfeito!"
            next_stage = "ASK_WHEN_TO_CALL"
            needs_ai = False

    # Tratamento especial para estágios específicos com fallbacks inteligentes
    elif stage == "ASK_EMAIL" and not ctx.email:
        # Tenta extrair email mesmo se NLU não detectou
        extracted_email = smart_email_extraction(user_text)
        if extracted_email:
            ctx.email = extracted_email
            response_text = f"Perfeito, anotei {extracted_email}!"
            next_stage = "ASK_WHEN_TO_CALL"
            needs_ai = False
        elif is_valid_email_attempt(user_text):
            # Parece tentativa de email, pede confirmação
            response_text = "Quase consegui anotar. Pode repetir seu email bem devagar? Por exemplo: joão ponto silva arroba gmail ponto com"
            needs_ai = False
            is_question = True

    elif stage == "ASK_NAME" and not ctx.user_name:
        # Tentativa mais agressiva de extrair nome
        clean_text = re.sub(r'^(meu nome é|me chamo|sou o|sou a|eu sou)\s*', '', user_text, flags=re.IGNORECASE).strip()
        words = clean_text.split()
        if words and len(words) <= 3 and all(word.isalpha() for word in words):
            potential_name = " ".join(words).title()
            ctx.user_name = potential_name
            response_text = f"Prazer, {potential_name}!"
            next_stage = "ASK_INTEREST"
            needs_ai = False

    # Tratamento para resistência/objeções
    elif intent in ("USER_RESISTANCE", "OBJECTION"):
        if "já tenho" in user_text.lower():
            response_text = "Entendo. Mesmo assim, vale comparar benefícios e preços. Posso mandar uma proposta sem compromisso?"
        else:
            response_text = "Sem problema. Quando seria melhor te ligar para uma conversa rápida?"
            next_stage = "ASK_WHEN_TO_CALL"
        needs_ai = False

    return response_text, is_question, needs_ai, next_stage

# Função para resetar contadores (útil para testes)
def reset_loop_detection():
    """Reseta contadores de detecção de loop"""
    global _question_repetition_count, _last_question_time
    _question_repetition_count.clear()
    _last_question_time.clear()