# -*- coding: utf-8 -*-
import asyncio
import random
import re
from typing import List, Dict, Optional, Tuple
import logging

log = logging.getLogger(__name__)

class ConversationState:
    """Mantém o estado da conversa do cliente simulado"""
    def __init__(self):
        self.name_given = False
        self.email_given = False
        self.product_chosen = False
        self.schedule_given = False
        self.conversation_turns = 0
        self.resistance_count = 0
        self.confusion_count = 0
        self.last_intent = None
        self.context_memory = []  # Lembra do que foi discutido

class CustomerPersonality:
    """Define comportamentos específicos para cada tipo de cliente"""
    
    PERSONALITIES = {
        "cooperativo": {
            "resistance_prob": 0.1,
            "confusion_prob": 0.05,
            "direct_answers": True,
            "patience": 5,
            "small_talk": False,
            "clarification_style": "formal"
        },
        "ocupado": {
            "resistance_prob": 0.4,
            "confusion_prob": 0.1,
            "direct_answers": True,
            "patience": 3,
            "small_talk": False,
            "clarification_style": "brief",
            "rush_indicators": ["rápido", "sem tempo", "ligação rápida"]
        },
        "cético": {
            "resistance_prob": 0.6,
            "confusion_prob": 0.15,
            "direct_answers": False,
            "patience": 4,
            "small_talk": True,
            "clarification_style": "questioning",
            "objections": ["já tenho seguro", "não preciso", "muito caro"]
        },
        "confuso": {
            "resistance_prob": 0.2,
            "confusion_prob": 0.4,
            "direct_answers": False,
            "patience": 6,
            "small_talk": True,
            "clarification_style": "repetitive"
        },
        "amigável": {
            "resistance_prob": 0.05,
            "confusion_prob": 0.08,
            "direct_answers": True,
            "patience": 7,
            "small_talk": True,
            "clarification_style": "chatty"
        }
    }
    
    def __init__(self, personality_type: str):
        self.type = personality_type
        self.config = self.PERSONALITIES.get(personality_type, self.PERSONALITIES["cooperativo"])

class ImprovedCustomerSimulator:
    """Simulador de cliente mais inteligente e variado"""
    
    def __init__(self, persona: str = "cooperativo", seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        
        # Normalização de persona
        persona_map = {
            "ocupado": "ocupado", "busy": "ocupado",
            "cooperativo": "cooperativo", "polite": "cooperativo", "educado": "cooperativo",
            "cético": "cético", "skeptical": "cético", "cetico": "cético",
            "confuso": "confuso", "confused": "confuso",
            "amigável": "amigável", "friendly": "amigável", "amigavel": "amigável"
        }
        
        self.persona_type = persona_map.get(persona.lower(), "cooperativo")
        self.personality = CustomerPersonality(self.persona_type)
        self.state = ConversationState()
        
        # Dados do cliente
        self.name = random.choice(["Carlos", "Ana", "João", "Maria", "Pedro", "Fernanda", "Ricardo", "Juliana"])
        self.email = f"{self.name.lower()}.{random.choice(['silva', 'santos', 'oliveira'])}@{random.choice(['gmail.com', 'hotmail.com', 'yahoo.com.br'])}"
        self.preferred_product = random.choice(["auto", "vida", "residencial", "viagem", "saúde"])
        
    async def generate_response(self, history: List[Dict[str, str]], expected_question: str = None) -> str:
        """Gera resposta baseada no contexto e personalidade"""
        
        self.state.conversation_turns += 1
        
        # Pega a última fala da Livia
        last_livia_message = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_livia_message = msg.get("text", "")
                break
        
        # Determina o que a Livia está pedindo
        intent = self._detect_livia_intent(last_livia_message)
        
        # Gera resposta baseada no intent e personalidade
        response = await self._generate_contextual_response(intent, last_livia_message, history)
        
        # Aplica variações de personalidade
        response = self._apply_personality_variations(response, intent)
        
        log.info(f"[{self.persona_type.upper()}] Intent: {intent} -> Response: '{response}'")
        
        return response
    
    def _detect_livia_intent(self, livia_message: str, expected: str = None) -> str:
        """Detecta o que a Livia está pedindo"""
        
        text = (livia_message or "").lower()
        
        # Padrões mais específicos e robustos
        name_patterns = [
            "nome", "como se chama", "qual seu nome", "posso saber seu nome",
            "me diga seu nome", "qual é o seu nome", "como você se chama"
        ]
        
        email_patterns = [
            "e-mail", "email", "endereço eletrônico", "melhor e-mail",
            "seu email", "qual seu email", "passa seu email"
        ]
        
        product_patterns = [
            "seguro", "interesse", "produto", "tipo de seguro", "qual seguro",
            "que seguro", "proteção", "cobertura"
        ]
        
        schedule_patterns = [
            "horário", "dia", "quando", "ligar", "retorno", "melhor horário",
            "que dia", "quando posso", "melhor dia"
        ]
        
        if any(pattern in text for pattern in name_patterns):
            return "asking_name"
        
        if any(pattern in text for pattern in email_patterns):
            return "asking_email"
        
        if any(pattern in text for pattern in product_patterns):
            return "asking_product"
        
        if any(pattern in text for pattern in schedule_patterns):
            return "asking_schedule"
        
        if any(pattern in text for pattern in ["tchau", "obrigad", "encerrar", "até mais"]):
            return "ending"
        
        # Detecta se Livia está repetindo pergunta (mais rigoroso)
        if self._is_livia_repeating_question(text):
            return "repeating_question"
        
        # Detecta cumprimentos/acknowledgments
        if any(pattern in text for pattern in ["boa tarde", "prazer", "perfeito", "ótimo"]):
            return "acknowledgment"
        
        return "general_conversation"
    
    def _is_livia_repeating_question(self, text: str) -> bool:
        """Detecta se Livia está repetindo a mesma pergunta"""
        # Verifica se tem frases repetitivas típicas
        repetitive_patterns = [
            "qual é o seu melhor e-mail",
            "próximos passos",
            "pode repetir",
            "mais uma vez"
        ]
        return any(pattern in text for pattern in repetitive_patterns)
    
    async def _generate_contextual_response(self, intent: str, livia_message: str, history: List[Dict]) -> str:
        """Gera resposta contextual baseada no intent"""
        
        # Conta quantas vezes já respondeu a pergunta similar
        similar_intents = sum(1 for record in self.state.context_memory if record.get("intent") == intent)
        
        if intent == "asking_name":
            return await self._respond_to_name_request(similar_intents)
        
        elif intent == "asking_email":
            return await self._respond_to_email_request(similar_intents)
        
        elif intent == "asking_product":
            return await self._respond_to_product_request(similar_intents)
        
        elif intent == "asking_schedule":
            return await self._respond_to_schedule_request(similar_intents)
        
        elif intent == "repeating_question":
            return await self._handle_repetitive_question(livia_message, history)
        
        elif intent == "acknowledgment":
            return await self._respond_to_acknowledgment(livia_message)
        
        elif intent == "ending":
            return self._generate_goodbye()
        
        else:
            return await self._generate_general_response(livia_message)
    
    async def _respond_to_acknowledgment(self, livia_message: str) -> str:
        """Responde a cumprimentos e acknowledgments da Livia"""
        
        if "prazer" in livia_message.lower():
            if self.persona_type == "amigável":
                return "O prazer é meu!"
            else:
                return "Igualmente."
        
        if any(word in livia_message.lower() for word in ["boa tarde", "bom dia", "boa noite"]):
            if self.persona_type == "ocupado":
                return "Oi. Vamos direto ao assunto?"
            elif self.persona_type == "amigável":
                return "Boa tarde! Tudo bem?"
            else:
                return "Boa tarde."
        
        # Acknowledgments genéricos
        return random.choice(["Ok.", "Certo.", "Entendi."])
    
    async def _respond_to_name_request(self, attempt: int) -> str:
        """Responde pedido de nome"""
        
        # Se já deu o nome e está sendo perguntado novamente
        if self.state.name_given and attempt > 0:
            return random.choice([
                f"Já disse, meu nome é {self.name}.",
                f"Como falei, sou {self.name}.",
                f"{self.name}, conforme mencionei."
            ])
        
        # Primeira vez sendo perguntado - aplica resistência apenas para céticos
        if not self.state.name_given and self.persona_type == "cético" and random.random() < 0.3:
            return random.choice([
                "Por que precisa do meu nome?",
                "Prefiro não dar meu nome ainda.",
                "Primeiro me diga sobre o que é."
            ])
        
        # Marca que já deu o nome
        self.state.name_given = True
        
        # Registra no contexto
        self.state.context_memory.append({"intent": "asking_name", "given": True})
        
        # Variações na forma de dar o nome
        if self.persona_type == "ocupado":
            return f"{self.name}. Pode ser rápido?"
        elif self.persona_type == "amigável":
            return f"Claro! Meu nome é {self.name}, prazer em falar com você."
        elif self.persona_type == "cético":
            return f"Tá bom, sou {self.name}."
        else:
            return random.choice([
                f"Sou {self.name}.",
                f"Me chamo {self.name}.",
                f"Meu nome é {self.name}."
            ])
    
    async def _respond_to_email_request(self, attempt: int) -> str:
        """Responde pedido de email"""
        
        # Se já deu o email antes
        if self.state.email_given and attempt > 0:
            return random.choice([
                f"Já informei: {self.email}",
                f"Como disse, é {self.email}",
                "Já passei meu email."
            ])
        
        # Confusão na primeira vez (alguns clientes)
        if attempt == 0 and random.random() < self.personality.config["confusion_prob"]:
            return random.choice([
                "Email? Para quê?",
                "Que email?",
                "Não entendi bem isso do email."
            ])
        
        # Resistência - apenas para céticos e ocupados
        if self.persona_type in ["cético", "ocupado"] and random.random() < self.personality.config["resistance_prob"] * 0.4:
            return random.choice([
                "Por que precisa do meu email?",
                "Prefiro não dar meu email agora.",
                "Primeiro quero saber mais detalhes."
            ])
        
        # Marca que deu o email e registra contexto
        self.state.email_given = True
        self.state.context_memory.append({"intent": "asking_email", "given": True})
        
        # Variações na forma de dar o email
        if self.persona_type == "ocupado":
            return f"{self.email}. Mais alguma coisa?"
        elif self.persona_type == "amigável":
            return f"Claro, meu email é {self.email}."
        elif self.persona_type == "cético":
            return f"Tá bom, é {self.email}."
        else:
            # Às vezes fala o email de forma mais "natural"
            if random.random() < 0.3:
                spoken_email = self.email.replace("@", " arroba ").replace(".", " ponto ")
                return f"Anota aí: {spoken_email}"
            else:
                return f"É {self.email}"
    
    async def _respond_to_product_request(self, attempt: int) -> str:
        """Responde sobre interesse em produtos"""
        
        if self.state.product_chosen and attempt > 0:
            return f"Como disse, tenho interesse em seguro {self.preferred_product}."
        
        # Alguns clientes não sabem bem o que querem
        if self.persona_type == "confuso" and random.random() < 0.4:
            return random.choice([
                "Não sei bem... que tipos vocês têm?",
                "Depende, o que vocês oferecem?",
                "Quais são as opções?"
            ])
        
        # Céticos questionam primeiro
        if self.persona_type == "cético" and random.random() < 0.3:
            return random.choice([
                "Já tenho seguro, por que mudaria?",
                "Que vantagem vocês oferecem?",
                "Qual a diferença do que já tenho?"
            ])
        
        self.state.product_chosen = True
        
        product_responses = {
            "auto": ["seguro do carro", "seguro automotivo", "para meu carro"],
            "vida": ["seguro de vida", "proteção para família"],
            "residencial": ["seguro da casa", "para minha residência"],
            "viagem": ["seguro viagem", "para viagens"],
            "saúde": ["plano de saúde", "seguro saúde"]
        }
        
        product_name = random.choice(product_responses[self.preferred_product])
        
        if self.persona_type == "amigável":
            return f"Tenho interesse em {product_name}. Pode me explicar as opções?"
        else:
            return f"Interesse em {product_name}."
    
    async def _respond_to_schedule_request(self, attempt: int) -> str:
        """Responde sobre horário para retorno"""
        
        if self.state.schedule_given and attempt > 0:
            return "Já disse quando prefiro que liguem."
        
        # Ocupados querem horários específicos
        if self.persona_type == "ocupado":
            times = ["manhã cedo", "depois das 18h", "sábado de manhã"]
            preferred_time = random.choice(times)
            return f"Só posso falar {preferred_time}. Pode ser?"
        
        # Flexíveis
        elif self.persona_type in ["cooperativo", "amigável"]:
            times = ["qualquer horário da tarde", "de manhã", "depois do almoço"]
            return f"Pode ser {random.choice(times)}."
        
        # Céticos
        elif self.persona_type == "cético":
            return "Primeiro quero ver se realmente me interessa. Pode mandar por email?"
        
        self.state.schedule_given = True
        return "Qualquer horário está bom."
    
    async def _handle_repetitive_question(self, livia_message: str, history: List[Dict]) -> str:
        """Lida com perguntas repetitivas da Livia"""
        
        self.state.confusion_count += 1
        
        # Cliente fica irritado após muitas repetições
        if self.state.confusion_count > 3:
            return random.choice([
                "Você já perguntou isso várias vezes!",
                "Não estou entendendo o que quer.",
                "Pode me transferir para outra pessoa?",
                "Acho melhor eu desligar e ligar depois."
            ])
        
        # Respostas baseadas na personalidade
        if self.persona_type == "ocupado":
            return random.choice([
                "Falei rápido, não deu para entender?",
                "Preciso que seja mais direto.",
                "Não tenho muito tempo, pode acelerar?"
            ])
        
        elif self.persona_type == "confuso":
            return random.choice([
                "Não entendi bem o que você quer.",
                "Pode explicar de outro jeito?",
                "Estou meio perdido aqui."
            ])
        
        elif self.persona_type == "cético":
            return random.choice([
                "Não está muito claro isso.",
                "Que informação exatamente você quer?",
                "Por que precisa disso?"
            ])
        
        else:  # cooperativo/amigável
            return random.choice([
                "Desculpe, pode repetir de outro jeito?",
                "Não ficou muito claro.",
                "Pode explicar melhor?"
            ])
    
    async def _generate_general_response(self, livia_message: str) -> str:
        """Gera resposta geral para outras situações"""
        
        # Respostas de acknowledgment
        if any(word in livia_message.lower() for word in ["perfeito", "ótimo", "certo", "obrigado"]):
            if self.persona_type == "amigável":
                return random.choice(["De nada!", "Que bom!", "Disponha!"])
            else:
                return random.choice(["Certo.", "Ok.", "Uhum."])
        
        # Resposta padrão
        return random.choice([
            "Certo.",
            "Entendi.",
            "Ok.",
            "Pode continuar."
        ])
    
    def _generate_goodbye(self) -> str:
        """Gera despedida"""
        return random.choice([
            "Obrigado, até mais!",
            "Valeu, tchau!",
            "Ok, obrigado pelo contato.",
            "Até mais!"
        ])
    
    def _apply_personality_variations(self, response: str, intent: str) -> str:
        """Aplica variações baseadas na personalidade"""
        
        # Adiciona ruído/fillers para algumas personalidades
        if self.persona_type == "confuso" and random.random() < 0.2:
            fillers = ["ahn...", "tipo...", "né..."]
            response = f"{random.choice(fillers)} {response}"
        
        # Ocupados são mais diretos
        elif self.persona_type == "ocupado" and random.random() < 0.3:
            response = response.replace(".", "").replace(",", "") + "."
        
        # Céticos questionam mais
        elif self.persona_type == "cético" and random.random() < 0.2:
            if not response.endswith("?"):
                questions = [" Por quê?", " Mas por quê?", " Tem certeza?"]
                response += random.choice(questions)
        
        return response
    
    # Método de compatibilidade com a API existente
    async def generate_reply(self, history: List[Dict[str, str]], expected_question: Optional[str] = None) -> str:
        return await self.generate_response(history, expected_question)