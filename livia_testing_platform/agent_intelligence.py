# agent_intelligence.py (re-escrito)

# -*- coding: utf-8 -*-
import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from improved_rules import smart_email_extraction

# Heurística de nome (fallback caso não exista em regras.py)
try:
    from regras import is_likely_name
except Exception:  # fallback simples
    def is_likely_name(s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        # 1 a 3 palavras com letras (inclui acentos)
        return bool(re.match(r"^[A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,2}$", s))

logger = logging.getLogger(__name__)

# ===== Regras/Regex base =====
_FORBIDDEN_RE = re.compile(
    r"\b(cot[aã]?[ç]?[aã]o|or[çc]amento|proposta|simula[ç]?[aã]o|"
    r"fechar|contratar|valor|pre[çc]o|mensalidade|parcelas?|desconto|promoção)\b",
    re.IGNORECASE,
)
_SENSITIVE_TOPICS = re.compile(
    r"\b(dados pessoais|cpf|rg|cartão|conta bancária|senha|pix)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)

# Sugestões de tempo (para reagendar etc.)
_WHEN_HINTS = [
    "amanhã", "amanha", "hoje",
    "de manhã", "de manha", "manhã", "manha",
    "de tarde", "tarde",
    "de noite", "noite",
    "cedo", "mais tarde"
]
_WEEKDAYS = r"(segunda|ter[aá]|quarta|quinta|sexta|s[áa]bado|sabado|domingo)"
_TIME_RE = re.compile(r"\b(\d{1,2})(?:[:h](\d{2}))?\b", re.I)

# ===== Política de e-mails da empresa =====
COMPANY_DOMAINS = tuple(d.strip().lower() for d in os.getenv(
    "COMPANY_EMAIL_DOMAINS",
    "cardinalle.com.br;cardinalleseguros.com.br"
).split(";"))

def _is_company_email(addr: str) -> bool:
    try:
        dom = addr.split("@", 1)[1].lower()
        return any(dom.endswith(cd) for cd in COMPANY_DOMAINS)
    except Exception:
        return False

# ===== Utilitários de texto =====

def _extract_when_phrase(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    low = t.lower()
    if any(k in low for k in _WHEN_HINTS):
        return t
    if re.search(_WEEKDAYS, low, re.I):
        return t
    if _TIME_RE.search(low):
        return t
    return None

def _dedupe_phrases(text: str) -> str:
    if not text:
        return text
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    out, seen = [], set()
    for p in parts:
        norm = re.sub(r'\s+', ' ', p).strip().lower().rstrip('.!?…')
        if norm and norm not in seen:
            seen.add(norm)
            out.append(p.strip())
    return " ".join(out)

def _sanitize_style(text: str, ctx: Optional[dict] = None) -> str:
    if not text:
        return text
    # neutro e direto
    text = re.sub(r"(?i)prazer em conhec[êe]-?l[oa]", "prazer em conhecer você", text)
    # corta meta-frases
    text = re.sub(r"(?i)poderia ser mais direto\??", "", text)
    text = re.sub(r"(?i)pode repetir a pergunta(, por favor)?\??", "", text)
    text = re.sub(r"(?i)entendo que .*?(?:\.\s*|$)", "", text)
    # tira interjeições e nomes quebrados
    text = re.sub(r"(?i)\b(bem|uhum|show|certo|então|entao|ok|beleza)\b[,\.…]*\s*", "", text)
    text = re.sub(r"(?i)\bmariana\s*[,\.]*", "Mariana", text)  # evita "Mariana.,"
    # pontuação e espaços
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

# ===== Guards de negócio =====

def _apply_business_guards(text: str, context: Optional[dict] = None) -> str:
    if not text:
        return text

    # Mascarar e-mails **da empresa** (não do cliente)
    def _mask(match):
        addr = match.group(0)
        return "[removido]" if _is_company_email(addr) else addr

    text = _EMAIL_RE.sub(_mask, text)

    # Só pedir e-mail se ainda NÃO temos e-mail válido do cliente
    ctx_email = (context or {}).get("email")
    if not ctx_email:
        if not re.search(r'(?i)\b(e-?mail|email)\b', text):
            text += " Prefiro que você me informe o seu e-mail para contato."

    # Conteúdo sensível / políticas
    if _FORBIDDEN_RE.search(text):
        return "Nosso especialista pode explicar melhor as opções e coberturas disponíveis."
    if _SENSITIVE_TOPICS.search(text):
        return "Por segurança, não tratamos dados sensíveis por telefone. Nosso especialista vai orientar você."
    return text

# ===== Fallback local determinístico =====

def _local_micro_answer(user_text: str, context: dict, knowledge: dict) -> str:
    """
    Resposta curta determinística quando o LLM falha, SEM loop.
    Sempre retorna frase breve + ">>> RETOMAR: <última pergunta>".
    """
    t = (user_text or "").lower().strip()
    # Proteção: se por engano vier prompt de sistema, limpa
    if "seu objetivo é" in t or "### contexto" in t or ">>> retomar" in t:
        t = ""

    short_answer = "Tudo certo."
    faq = knowledge.get("faq", {})

    if "como você se chama" in t or "seu nome" in t:
        short_answer = faq.get("meu_nome", "Meu nome é Livia.")
    elif "cardinalle" in t or "sua empresa" in t or "quem são vocês" in t:
        short_answer = faq.get("sobre_a_cardinalle", "Somos a Cardinalle Seguros.")

    asked_questions = context.get("asked_questions", []) if isinstance(context, dict) else []
    last_question = asked_questions[-1] if asked_questions else "Para começar, qual é o seu nome, por favor?"
    return f"{short_answer}\n>>> RETOMAR: {last_question}"

# ===== Pós-processamento principal =====

def postprocess_and_ensure_retake(response_text: str, context: dict) -> str:
    try:
        RETAKE_MARK = ">>> RETOMAR:"
        cleaned = (response_text or "").strip()

        cleaned = _apply_business_guards(cleaned, context)
        cleaned = _sanitize_style(cleaned, context)
        cleaned = _dedupe_phrases(cleaned)

        if RETAKE_MARK in cleaned:
            return cleaned

        stage = ""
        if isinstance(context, dict):
            stage = str(context.get("stage", "")).upper()
        if stage == "END":
            return cleaned

        asked = (context or {}).get("asked_questions") or []
        last_q = asked[-1] if asked else None
        if last_q:
            return f"{cleaned}\n{RETAKE_MARK} {last_q}"
        else:
            return f"{cleaned}\n{RETAKE_MARK} NENHUMA"

    except Exception as e:
        logger.exception("Erro em postprocess_and_ensure_retake: %s", e)
        base = (response_text or "").strip()
        return f"{base}\n>>> RETOMAR: NENHUMA"

# ===== Agente principal =====

class LiviaTurbinadaAgent:
    def __init__(self, client, model: str, base_url: str, api_key: str, fallback_client: Optional[Any] = None):
        self.client = client
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key or ""
        self.fallback_client = fallback_client
        self.knowledge = self._load_knowledge()
        self.product_patterns = {
            r"(?i)\b(carro|autom[óo]vel|ve[íi]culo)\b": "auto",
            r"(?i)\b(vida|seguro de vida)\b": "vida",
            r"(?i)\b(casa|resid[êe]ncia|im[óo]vel)\b": "residencial",
            r"(?i)\b(viagem|travel)\b": "viagem",
            r"(?i)\b(sa[úu]de|plano de sa[úu]de)\b": "saúde",
            r"(?i)\b(previd[êe]ncia)\b": "previdência",
        }
        self.SYSTEM_PROMPT = (
            "Você é Livia, assistente de seguros da Cardinalle. "
            "Use linguagem neutra (você), evite 'conhecê-lo/conhecê-la'. "
            "Seja breve (1-2 frases), objetiva e SEM meta-frases (ex.: 'poderia ser mais direto?'). "
            "Nunca forneça e-mails/telefones da empresa; sempre peça o e-mail do cliente. "
            "Se o cliente disser 'liga mais tarde', 'sem tempo' ou equivalentes, avance para pedir dia/horário para retorno. "
            "Se houver interesse declarado (ex.: previdência, vida, auto), peça o e-mail para contato."
        )
        logger.info("LiviaTurbinadaAgent inicializado com LLM: %s", self.model)

    def _load_knowledge(self) -> Dict[str, Any]:
        logger.info("Usando conhecimento padrão embutido.")
        return {
            "agent": {"name": "Livia", "company": "Cardinalle", "role": "Consultora de seguros"},
            "products": {
                "auto": {"description": "Proteção para seu veículo contra roubo, colisão e danos."},
                "vida": {"description": "Segurança financeira para você e sua família."},
                "home": {"description": "Cobertura para sua casa contra incêndio, roubo e mais."},
                "saude": {"description": "Planos de saúde para consultas, exames e internações."},
                "travel": {"description": "Proteção para viagens, incluindo emergências."},
                "previdencia": {"description": "Planejamento para sua aposentadoria."},
            },
            "faq": {
                "meu_nome": "Meu nome é Livia.",
                "sobre_a_cardinalle": "A Cardinalle Seguros é uma consultoria experiente que ajuda clientes a encontrar a melhor proteção financeira.",
                "proximos_passos": "O próximo passo é uma conversa com um de nossos especialistas para entender suas necessidades.",
                "custo": "Não falamos de valores neste primeiro contato. Um especialista montará uma proposta personalizada.",
            },
            "intents": ["GIVE_NAME", "GIVE_EMAIL", "CHOOSE_PRODUCT", "USER_RESISTANCE", "AGREE", "OTHER"],
        }

    async def _call_llm(self, prompt: str, context: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 400,
        }
        headers_primary = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            }
        try:
            if not self.base_url:
                raise RuntimeError("BASE_URL ausente")
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers_primary,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Erro no LLM primário ({self.model}): {e}")
            # Fallback OpenAI (opcional)
            if self.fallback_client and os.getenv("OPENAI_API_KEY"):
                try:
                    fb_payload = {
                        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 400,
                    }
                    fb_headers = {
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json",
                    }
                    fb = await self.fallback_client.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=fb_payload,
                        headers=fb_headers,
                        timeout=30.0,
                    )
                    fb.raise_for_status()
                    data = fb.json()
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    logger.warning("Todos os LLMs falharam. Usando resposta local determinística.")
            # Fallback local — usa TEXTO REAL do usuário
            user_text = (context or {}).get("last_user_text", "")
            return _local_micro_answer(user_text, context=context or {}, knowledge=self.knowledge)

    def normalize_stage(self, stage_raw: str) -> str:
        if not stage_raw:
            return ""
        s = str(stage_raw).upper().replace("ESTAGIO_", "").replace("ESTÁGIO_", "")
        mapping = {
            "AWAITING_NAME": "ASK_NAME", "AWAIT_NAME": "ASK_NAME",
            "RESCHEDULE_ASK_NAME": "ASK_NAME",
        }
        return mapping.get(s, s)

    async def process_user_response(self, user_text: str, history: List[dict], context: Dict[str, Any], nlu_result: Dict[str, Any]) -> str:
        # Salva último texto do usuário para o fallback local
        try:
            if isinstance(context, dict):
                context["last_user_text"] = user_text
        except Exception:
            pass

        system_logic = (
            "Você é Livia, uma assistente de seguros. Intervenha de forma curta e direta "
            "e em seguida retome o roteiro com a última pergunta."
        )
        stage = context.get("stage") or context.get("Stage") or "N/A"
        asked_questions = context.get("asked_questions", [])
        last_question = asked_questions[-1] if asked_questions else "Para começar, qual é o seu nome, por favor?"

        full_prompt = f"""
{system_logic}

--- CONTEXTO ATUAL ---
Estágio: {stage}
Última pergunta do roteiro: "{last_question}"
Fala do usuário: "{user_text}"
----------------------

Responda em 1-2 frases e finalize repetindo a pergunta de retomada, se necessário.
""".strip()

        ai_text = await self._call_llm(full_prompt, context)
        return postprocess_and_ensure_retake(ai_text, context)

    async def nlu_router(self, user_input: str, stage: str, last_question: str) -> dict:
        stage_norm = self.normalize_stage(stage)
        text = (user_input or "").lower().strip()
        bye_keywords = ["tchau", "até mais", "valeu", "encerrar", "desligar"]
        if any(keyword in text for keyword in bye_keywords):
            return {"intent": "END_CONVERSATION", "confidence": 0.95}
        if not text:
            return {"intent": "OTHER", "confidence": 0.1}

        if any(w in text for w in ["obrigado", "valeu", "tá bom", "ta bom", "blz"]):
            return {"intent": "ACKNOWLEDGMENT", "confidence": 0.95}

        if "@" in text and "." in text:
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.[\w]{2,}", text)
            if email_match:
                return {"intent": "GIVE_EMAIL", "email": email_match.group(0), "confidence": 0.95}

        for pattern, product in self.product_patterns.items():
            if re.search(pattern, text):
                return {"intent": "CHOOSE_PRODUCT", "product": product, "confidence": 0.9}

        if stage_norm == "ASK_NAME":
            negative_words = ["não", "nao", "nada", "nenhum", "agora não", "agora nao"]
            tokens = text.split()
            if not any(w in text for w in negative_words) and 0 < len(tokens) <= 6:
                clean_text = re.sub(r'^(meu nome é|me chamo|eu sou|sou o|sou a|sou)\s*', '', user_input, flags=re.IGNORECASE).strip()
                if is_likely_name(cleaned):
                    return {"intent": "GIVE_NAME", "name": cleaned.title(), "confidence": 0.9}

        busy_kw = [
            "liga mais tarde","ligar mais tarde","fala depois","falar depois",
            "sem tempo","agora não dá","agora nao da","não posso falar agora","nao posso falar agora",
            "me liga depois","me ligar depois","retorna mais tarde","retorna depois"
        ]
        if any(k in text for k in busy_kw):
            return {"intent": "RESCHEDULE_REQUEST", "confidence": 0.9}

        when_phrase = _extract_when_phrase(text)
        if when_phrase:
            return {"intent": "GIVE_WHEN", "when": when_phrase, "confidence": 0.9}

        return {"intent": "OTHER", "confidence": 0.1}

