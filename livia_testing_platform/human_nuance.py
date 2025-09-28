# -*- coding: utf-8 -*-
"""
human_nuance.py — Heurísticas leves de "afeto" e renderização natural
"""
import os
import re
import random
from typing import Dict, Any
import logging
log = logging.getLogger(__name__)
AFFECTS = (
    "neutral", "friendly", "empathetic", "apologetic",
    "excited", "assertive", "professional", "relieved"
)
ACKS = ["Uhum", "Certo", "Perfeito", "Entendi", "Tá", "Claro", "Pode ser", "Show"]
CHECKS = ["pode ser?", "tudo bem?", "beleza?", "faz sentido?", "te ajuda?", "combinado?"]
ACK_WORDS = r"(?:claro|show|perfeito|uhum|certo|entendi|beleza|ok|okay|tá|ta|pode ser|maravilha|legal)"
LEAD_ACK_SINGLE = re.compile(rf"^\s*{ACK_WORDS}\s*[,:-]?\s*", re.IGNORECASE)
LEAD_ACK_MULTI = re.compile(rf"^\s*(?:{ACK_WORDS}\s*[,:-]?\s*){{2,}}", re.IGNORECASE)
END_CHECK_WORDS = r"(?:combinado|beleza|pode ser|tudo bem|faz sentido|te ajuda)"
END_CHECK_PRESENT = re.compile(rf"\s*{END_CHECK_WORDS}\s*\?$", re.IGNORECASE)
def affect_from_text(user_text: str) -> str:
    if not user_text:
        return "neutral"
    t = user_text.lower().strip()
    neg = any(k in t for k in ["não", "nao", "agora não", "ocupado", "depois", "caro", "ruim"])
    excite = "!" in t or any(k in t for k in ["ótimo", "otimo", "incrível", "maravilha", "top", "show", "perfeito", "legal"])
    is_question = "?" in t
    if any(end in t for end in ["obrigado", "tchau"]):
        return "relieved"
    if neg:
        return "empathetic"
    if len(t) < 18 and is_question:
        return "professional"
    if excite:
        return "friendly" if random.random() < 0.7 else "excited"
    return "neutral"
def _maybe(options, p=0.5):
    if not options or random.random() > p:
        return ""
    return random.choice(options)
def _collapse_leading_acks(text: str) -> str:
    if LEAD_ACK_MULTI.match(text or ""):
        m = LEAD_ACK_SINGLE.match(text or "")
        first = (m.group(0).strip(" ,:-") if m else "")
        rest = LEAD_ACK_MULTI.sub("", text, count=1).lstrip()
        return f"{first}, {rest}" if first else rest
    return text
def _punctuate(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"\s+([?!.,])", r"\1", t)
    if not t:
        return t
    if t and t[-1] not in ".?!…":
        t += "."
    return t
def voice_settings_for_affect(affect: str) -> Dict:
    base = {
        "stability": 0.38,
        "style": 0.60,
        "similarity_boost": 0.90,
        "use_speaker_boost": True,
    }
    table = {
        "neutral": {"stability": 0.40, "style": 0.60},
        "friendly": {"stability": 0.30, "style": 0.80},
        "empathetic": {"stability": 0.50, "style": 0.40},
        "apologetic": {"stability": 0.55, "style": 0.25},
        "excited": {"stability": 0.25, "style": 0.90},
        "assertive": {"stability": 0.40, "style": 0.70},
        "professional": {"stability": 0.45, "style": 0.55},
        "relieved": {"stability": 0.35, "style": 0.65},
    }
    sel = affect if affect in table else "neutral"
    return {**base, **table[sel]}
def render_natural(core_text: str, dialogue_context: Any, affect: str = "neutral", *, opening: bool = False) -> str:
    log.debug(f"Renderizando: core_text='{core_text}', affect={affect}, opening={opening}")
    prefix = ""
    if not dialogue_context:
        log.warning("dialogue_context é None, usando defaults")
        return _punctuate(core_text)
    if hasattr(dialogue_context, 'user_name') and dialogue_context.user_name and random.random() < 0.3:
        if dialogue_context.user_name.lower() not in core_text.lower():
            core_text = f"{dialogue_context.user_name}, {core_text[0].lower() + core_text[1:]}"
    if opening:
        txt = re.sub(LEAD_ACK_SINGLE, "", core_text, count=1)
        txt = re.sub(END_CHECK_PRESENT, "", txt)
        core_text = _punctuate(txt)
    if os.getenv("NUANCE_NO_FILLERS", "0") == "1":
        prefix = ""
    else:
        STRONG_OPENERS = re.compile(r'^(prazer|perfeito|certo|ok|que bom|desculpe|entendi)\b', re.IGNORECASE)
        skip_prefix = bool(STRONG_OPENERS.match(core_text))
        if not skip_prefix:
            ack_p_base = float(os.getenv("NUANCE_ACK_P", "0.18"))
            if affect == "friendly":
                prefix = _maybe(["Claro", "Perfeito", "Beleza", "Show"], p=ack_p_base)
            elif affect in ("empathetic", "apologetic"):
                prefix = _maybe(["Entendi", "Certo", "Uhum"], p=ack_p_base)
            else:
                prefix = _maybe(ACKS, p=ack_p_base)
            if prefix and LEAD_ACK_SINGLE.match(core_text):
                prefix = ""
    suffix = ""
    if (hasattr(dialogue_context, 'contact_time') and dialogue_context.contact_time and
        not any(neg in core_text.lower() for neg in ["não", "nao", "obrigado", "tchau"]) and
        not (hasattr(dialogue_context, 'stage') and dialogue_context.stage == "END")):
        if random.random() < 0.6:
            suffix = " Alguma dúvida antes de prosseguir?"
    if (os.getenv("NUANCE_NO_FILLERS", "0") != "1" and len(core_text.split()) > 8 and
        random.random() < 0.12 and affect not in ["professional", "apologetic"]):
        fillers = ["Hum...", "Bem...", "Então..."]
        if prefix:
            prefix += ", "
        else:
            prefix = random.choice(fillers) + " "
    out = (prefix + (", " if prefix and not prefix.endswith(", ") else "") + core_text + (suffix if suffix else "")).strip()
    final = _punctuate(out)
    if any(end in final.lower() for end in ["até mais", "obrigado", "tenha um ótimo dia"]):
        log.debug("Ajustando affect para relieved em encerramento")
    return final