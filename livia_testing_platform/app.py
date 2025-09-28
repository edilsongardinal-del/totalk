
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True  # evita travas em compilação de .pyc (ex.: idna)

import os
import asyncio
import logging
import re
import random

from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from agent_intelligence import LiviaTurbinadaAgent
from improved_orchestration import ImprovedOrchestrator
from regras import rules_handler_func, get_default_question_for_stage
from human_nuance import affect_from_text, render_natural
from improved_customer_simulator import ImprovedCustomerSimulator

os.environ.setdefault("NUANCE_NO_FILLERS", "1")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
SIM_RESULTS: List[Dict[str, Any]] = []

# === Helpers assíncronos ===

def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        raise

# === Inicialização do agente/LLM ===

def _init_agent() -> LiviaTurbinadaAgent:
    import httpx  # lazy import evita travas na inicialização

    base = os.getenv("PRIMARY_LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.getenv("PRIMARY_LLM_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("PRIMARY_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")

    default_headers = {
        "Accept": "application/json",
        # força a NÃO usar brotli ("br")
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "LiviaTesting/1.0 httpx",
    }

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=30.0, read=30.0, write=30.0),
        headers=default_headers,
        http2=False,
    )

    fallback_client = (
        httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=30.0, read=30.0, write=30.0),
            headers=default_headers,
            http2=False,
        ) if os.getenv("OPENAI_API_KEY") else None
    )

    return LiviaTurbinadaAgent(
        client=client,
        model=model,
        base_url=base,
        api_key=api_key,
        fallback_client=fallback_client,
    )

# === Orquestrador ===

def _init_orchestrator(agent: LiviaTurbinadaAgent) -> ImprovedOrchestrator:
    orch = ImprovedOrchestrator(agent, rules_handler_func=rules_handler_func)
    try:
        orch.context.stage = "ASK_NAME"
    except Exception:
        pass
    return orch

# === Utilidades de UI/Histórico ===

def _dedupe_sentences(text: str) -> str:
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


def _apply_retake_and_register_question(text: str, orchestrator):
    tag = ">>> RETOMAR:"
    core = text
    next_q = ""

    if tag in text:
        core, retake = text.split(tag, 1)
        next_q = (retake or "").strip()

    core = core.strip()

    if next_q and next_q.upper() != "NENHUMA" and getattr(orchestrator.context, "stage", "") != "END":
        if not getattr(orchestrator.context, "asked_questions", None):
            orchestrator.context.asked_questions = []
        if not orchestrator.context.asked_questions or orchestrator.context.asked_questions[-1] != next_q:
            orchestrator.context.asked_questions.append(next_q)
        orchestrator.context.last_question = next_q

    return text.replace(tag, "").strip()
    
def _collect_variables(ctx) -> Dict[str, Any]:
    return {
        "Nome": getattr(ctx, "user_name", "") or "",
        "Email": getattr(ctx, "email", "") or "",
        "Seguro": getattr(ctx, "insurance_type", "") or "",
        "Horario": getattr(ctx, "reschedule_time", "") or "",
        "Stage": getattr(ctx, "stage", "") or "",
    }


def _as_transcript(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for h in history:
        role = h.get("role")
        sender = "Livia" if role == "assistant" else "Cliente"
        out.append({"sender": sender, "text": h.get("text", "")})
    return out

# === Views ===
@app.get("/")
def dashboard():
    return render_template("dashboard.html")

@app.get("/simulator")
def simulator_page():
    return render_template("simulator.html")

@app.get("/success-analysis")
def success_analysis_page():
    return render_template("success_analysis.html")

@app.get("/rules-vs-ai")
def rules_vs_ai_page():
    return render_template("rules_vs_ai.html")

# === APIs ===
@app.get("/api/dashboard_data")
def dashboard_data():
    total = len(SIM_RESULTS)
    if total == 0:
        return jsonify({
            "total_conversations": 0,
            "success_rate": 0,
            "variable_collection": {"Nome": 0, "Email": 0, "Seguro": 0, "Horario": 0}
        })
    got = {"Nome": 0, "Email": 0, "Seguro": 0, "Horario": 0}
    success = 0
    for r in SIM_RESULTS:
        vc = r.get("variables_collected", {})
        all_ok = True
        for k in got:
            has_val = bool(vc.get(k))
            got[k] += 1 if has_val else 0
            all_ok = all_ok and has_val
        if all_ok:
            success += 1
    variable_collection = {k: round(100.0 * v / total, 1) for k, v in got.items()}
    success_rate = round(100.0 * success / total, 1)
    return jsonify({
        "total_conversations": total,
        "success_rate": success_rate,
        "variable_collection": variable_collection
    })

@app.get("/api/analysis_data")
def analysis_data():
    if not SIM_RESULTS:
        return jsonify({"matrix": {}, "failure_reasons": {}})
    matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: {"Nome": 0, "Email": 0, "Seguro": 0, "Horario": 0, "_count": 0})
    failures = Counter()
    for r in SIM_RESULTS:
        persona = r.get("persona", "desconhecida")
        vc = r.get("variables_collected", {})
        matrix[persona]["_count"] += 1
        for k in ("Nome", "Email", "Seguro", "Horario"):
            matrix[persona][k] += 1 if vc.get(k) else 0
        if r.get("end_reason") == "user_hangup":
            failures["Cliente desligou / Incompleto"] += 1
        else:
            if not vc.get("Email"):
                failures["Validação de formato de e-mail"] += 1
            elif not vc.get("Nome"):
                failures["Erro na extração de nome"] += 1
            else:
                failures["Outro / Incompleto"] += 1
    for p, row in matrix.items():
        total = max(1, row["_count"])
        for k in ("Nome", "Email", "Seguro", "Horario"):
            row[k] = round(100.0 * row[k] / total, 1)
        row.pop("_count", None)
    return jsonify({"matrix": matrix, "failure_reasons": dict(failures)})

@app.post("/api/start_simulation")
def start_simulation():
    payload = request.get_json(force=True, silent=True) or {}
    personality = (payload.get("personality") or "cooperativo").lower().strip()
    max_turns = int(payload.get("max_turns") or 15)

    # Cliente simulado melhorado - SEM seed para mais variação
    customer = ImprovedCustomerSimulator(persona=personality, seed=None)
    agent = _init_agent()
    orchestrator = _init_orchestrator(agent)

    history: List[Dict[str, str]] = []

    async def run_dialogue() -> Dict[str, Any]:
        try:
            opening_q = get_default_question_for_stage(orchestrator.context)
        except Exception:
            opening_q = "Para começar, qual é o seu nome, por favor?"
        orchestrator.context.last_question = opening_q
        if not getattr(orchestrator.context, "asked_questions", None):
            orchestrator.context.asked_questions = []
        orchestrator.context.asked_questions.append(opening_q)

        # Abertura mais natural
        greeting = "Boa tarde, aqui é a Livia, da Cardinalle Seguros."
        livia_opening = f"{greeting} {opening_q}"
        history.append({"role": "assistant", "text": livia_opening})

        end_reason = "max_turns"
        consecutive_confusion = 0
        last_responses = []

        for turn in range(max_turns):
            # === Cliente responde AO TEXTO (sem expected oculto) ===
            try:
                customer_text = await customer.generate_response(history, expected_question=None)
            except Exception as e:
                log.error(f"Erro na geração de resposta do cliente: {e}")
                customer_text = "Desculpe, não entendi."
            if not customer_text or customer_text.strip() == "":
                customer_text = "Certo."
            history.append({"role": "user", "text": customer_text})

            # === NLU ===
            expected = getattr(orchestrator.context, "last_question", None) or ""
            try:
                nlu = await agent.nlu_router(customer_text, getattr(orchestrator.context, "stage", "ASK_NAME"), expected)
            except Exception:
                logging.exception("Erro no NLU router")
                nlu = {"intent": "OTHER", "confidence": 0.1}

            try:
                orchestrator.context.update_from_nlu(nlu)
                orchestrator.context.update_output_json(nlu)
            except Exception:
                logging.exception("Falha ao aplicar NLU no contexto")

            # === Regras ===
            response_text = ""
            next_stage = getattr(orchestrator.context, "stage", "ASK_NAME")
            needs_ai = True
            is_question = False
            try:
                response_text, is_question, needs_ai, next_stage = await rules_handler_func(
                    orchestrator, customer_text, nlu, history, next_stage, expected
                )
            except Exception:
                logging.exception("Erro nas regras")
                response_text = ""
                needs_ai = True

            if next_stage:
                orchestrator.context.stage = next_stage

            # === Planejar próxima pergunta ===
            try:
                planned_q = get_default_question_for_stage(orchestrator.context) if getattr(orchestrator.context, "stage", "") != "END" else None
            except Exception:
                planned_q = None
            if planned_q:
                if not getattr(orchestrator.context, "asked_questions", None):
                    orchestrator.context.asked_questions = []
                if not orchestrator.context.asked_questions or orchestrator.context.asked_questions[-1] != planned_q:
                    orchestrator.context.asked_questions.append(planned_q)
                orchestrator.context.last_question = planned_q

            # === IA (ou fallback) ===
            if needs_ai or not response_text:
                try:
                    ai_text = await agent.process_user_response(
                        user_text=customer_text,
                        history=history,
                        context=orchestrator.context.__dict__,
                        nlu_result=nlu,
                    )
                    response_text = ai_text
                except Exception as e:
                    logging.getLogger().error("Erro no LLM: %s", e)
                    if orchestrator.context.stage == "ASK_NAME":
                        response_text = "Para começarmos, qual é o seu nome?"
                    elif orchestrator.context.stage == "ASK_EMAIL":
                        response_text = "Qual é o seu melhor e-mail para contato?"
                    elif orchestrator.context.stage == "ASK_INTEREST":
                        response_text = "Que tipo de seguro te interessa?"
                    else:
                        response_text = "Certo. Como posso te ajudar?"

            # === Aplicar >>> RETOMAR e anexar pergunta ===
            response_text = _apply_retake_and_register_question(response_text, orchestrator)

            # Evita respostas muito repetitivas
            if response_text in last_responses[-2:]:
                variations = [
                    f"Como mencionei, {response_text.lower()}",
                    f"Para esclarecer: {response_text}",
                    f"Voltando à pergunta: {response_text}",
                ]
                response_text = random.choice(variations)
            last_responses.append(response_text)
            if len(last_responses) > 5:
                last_responses.pop(0)

            # === Humanização e push no histórico ===
            af = affect_from_text(customer_text)
            livia_text = render_natural(response_text, orchestrator.context, affect=af, opening=False)
            history.append({"role": "assistant", "text": livia_text})

            # === Critérios de término ===
            if getattr(orchestrator.context, "stage", "") == "END":
                end_reason = "conversation_complete"
                break

            hangup_phrases = ["desligar", "outro dia", "não tenho tempo", "tchau", "até mais", "valeu"]
            if any(phrase in customer_text.lower() for phrase in hangup_phrases):
                end_reason = "customer_hangup"
                break

            if (
                orchestrator.context.user_name and
                (orchestrator.context.email or orchestrator.context.reschedule_time) and
                orchestrator.context.insurance_type
            ):
                end_reason = "success_complete"
                break

            if turn > 8 and not any([
                orchestrator.context.user_name,
                orchestrator.context.email,
                orchestrator.context.reschedule_time,
            ]):
                end_reason = "no_progress"
                break

        vars_collected = _collect_variables(orchestrator.context)
        conversation_analysis = analyze_conversation_quality(history, vars_collected)

        return {
            "persona": personality,
            "history": history,
            "variables_collected": vars_collected,
            "end_reason": end_reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_turns": len([m for m in history if m.get("role") == "user"]),
            "personality_metrics": {
                "confusion_episodes": consecutive_confusion,
                "customer_name": customer.name,
                "customer_email": customer.email,
                "preferred_product": customer.preferred_product,
            },
            "quality_analysis": conversation_analysis,
        }

    result = _run_async(run_dialogue())
    SIM_RESULTS.append(result)

    transcript = _as_transcript(result["history"])
    return jsonify({
        "transcript": transcript,
        "variables_collected": result["variables_collected"],
        "end_reason": result["end_reason"],
        "persona": result["persona"],
        "timestamp": result["timestamp"],
        "total_turns": result.get("total_turns", 0),
        "personality_metrics": result.get("personality_metrics", {}),
        "quality_analysis": result.get("quality_analysis", {}),
    })

# === Métricas de qualidade ===

def analyze_conversation_quality(history: List[Dict], variables: Dict) -> Dict:
    user_messages = [m for m in history if m.get("role") == "user"]
    assistant_messages = [m for m in history if m.get("role") == "assistant"]

    if not user_messages:
        return {"quality_grade": "F - Sem Interação", "total_turns": 0}

    total_turns = len(user_messages)
    avg_user_length = sum(len(m.get("text", "").split()) for m in user_messages) / max(1, len(user_messages))

    assistant_texts = [m.get("text", "") for m in assistant_messages]
    unique_assistant_responses = len(set(assistant_texts))
    repeated_questions = len(assistant_texts) - unique_assistant_responses

    completion_score = sum(1 for v in variables.values() if v) / len(variables)

    confusion_phrases = ["não entendi", "pode ser mais direto", "como assim", "pode repetir"]
    confused_responses = sum(1 for msg in user_messages if any(phrase in msg.get("text", "").lower() for phrase in confusion_phrases))
    naturalness_score = max(0, (len(user_messages) - confused_responses) / max(1, len(user_messages)))

    stages_mentioned = sum(1 for v in [variables.get("Nome"), variables.get("Email"), variables.get("Seguro")] if v)
    progression_score = stages_mentioned / 3.0

    return {
        "total_turns": total_turns,
        "avg_user_message_length": round(avg_user_length, 1),
        "repeated_questions": repeated_questions,
        "completion_rate": round(completion_score * 100, 1),
        "naturalness_score": round(naturalness_score * 100, 1),
        "progression_score": round(progression_score * 100, 1),
        "quality_grade": _calculate_quality_grade(completion_score, naturalness_score, repeated_questions, progression_score),
    }


def _calculate_quality_grade(completion: float, naturalness: float, repeated: int, progression: float) -> str:
    base_score = (completion * 0.3 + naturalness * 0.4 + progression * 0.3) * 100
    penalty = min(repeated * 15, 40)
    final_score = max(0, base_score - penalty)

    if final_score >= 85:
        return "A - Excelente"
    elif final_score >= 70:
        return "B - Boa"
    elif final_score >= 55:
        return "C - Regular"
    elif final_score >= 40:
        return "D - Fraca"
    else:
        return "F - Muito Fraca"

# === Utilitários diversos ===

@app.post("/api/reset_simulations")
def reset_simulations():
    global SIM_RESULTS
    SIM_RESULTS = []
    return jsonify({"message": "Dados de simulação resetados", "status": "success"})

@app.get("/api/detailed_stats")
def detailed_stats():
    if not SIM_RESULTS:
        return jsonify({"error": "Nenhuma simulação disponível"})

    stats = {
        "total_simulations": len(SIM_RESULTS),
        "by_persona": {},
        "by_end_reason": Counter(),
        "quality_distribution": Counter(),
        "avg_turns": 0,
        "avg_completion_rate": 0,
    }

    total_turns = 0
    total_completion = 0

    for result in SIM_RESULTS:
        persona = result.get("persona", "unknown")
        if persona not in stats["by_persona"]:
            stats["by_persona"][persona] = {"count": 0, "success_rate": 0, "avg_quality": 0}
        stats["by_persona"][persona]["count"] += 1

        stats["by_end_reason"][result.get("end_reason", "unknown")] += 1

        quality = result.get("quality_analysis", {})
        grade = quality.get("quality_grade", "F - Muito Fraca")
        stats["quality_distribution"][grade] += 1

        total_turns += result.get("total_turns", 0)
        total_completion += quality.get("completion_rate", 0)

    stats["avg_turns"] = round(total_turns / len(SIM_RESULTS), 1)
    stats["avg_completion_rate"] = round(total_completion / len(SIM_RESULTS), 1)

    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5001")))
