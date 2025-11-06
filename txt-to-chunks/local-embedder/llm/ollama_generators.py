from typing import List

from llama_index.core import Document
from llama_index.llms.ollama import Ollama

from core.qa import QuestionGenerator, AnswerGenerator


class LLMQuestionGenerator(QuestionGenerator):
    """Générateur de questions utilisant un LLM via Ollama."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct", temperature: float = 0.1, timeout: float = 60.0) -> None:
        self.llm = Ollama(model=llm_model, temperature=temperature, request_timeout=timeout)
    
    def generate_questions(self, chunk: Document) -> List[str]:
        prompt = f"""
Basé sur le texte suivant, génère 2-3 questions en français pertinentes qui pourraient être posées par un étudiant en droit administratif.

Texte:
{chunk.text}

Génère des questions uniquement en français qui:
1. Testent la compréhension des concepts clés
2. Demandent des définitions importantes
3. Explorent les différences entre concepts
4. Sont spécifiques au contenu du texte

Format: Une question par ligne, sans numérotation.
"""
        response = self.llm.complete(prompt)
        questions_text = str(response)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        questions = [q for q in questions if len(q) > 10 and '?' in q]
        return questions[:3]


class LLMAnswerGenerator(AnswerGenerator):
    """Générateur de réponses utilisant un LLM via Ollama."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct", temperature: float = 0.1, timeout: float = 60.0) -> None:
        self.llm = Ollama(model=llm_model, temperature=temperature, request_timeout=timeout)
    
    def generate_answer(self, question: str, chunk: Document) -> str:
        prompt = f"""
Tu es un assistant spécialisé en droit administratif. Réponds en français à la question suivante en te basant uniquement sur le texte fourni.

Question: {question}

Texte de référence:
{chunk.text}

Instructions:
1. Réponds uniquement en français
2. Utilise uniquement les informations du texte fourni
3. Sois précis et concis
4. Si la réponse n'est pas dans le texte, dis-le clairement
5. Structure ta réponse de manière claire

Réponse:
"""
        response = self.llm.complete(prompt)
        return str(response).strip()

from typing import List

from llama_index.core import Document
from llama_index.llms.ollama import Ollama

from core.qa import QuestionGenerator, AnswerGenerator


class LLMQuestionGenerator(QuestionGenerator):
    """Générateur de questions utilisant un LLM (Ollama)."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct", temperature: float = 0.1, request_timeout: float = 60.0):
        self.llm = Ollama(model=llm_model, temperature=temperature, request_timeout=request_timeout)
    
    def generate_questions(self, chunk: Document) -> List[str]:
        prompt = f"""
Basé sur le texte suivant, génère 2-3 questions en français pertinentes qui pourraient être posées par un étudiant en droit administratif.

Texte:
{chunk.text}

Génère des questions uniquement en français qui:
1. Testent la compréhension des concepts clés
2. Demandent des définitions importantes
3. Explorent les différences entre concepts
4. Sont spécifiques au contenu du texte

Format: Une question par ligne, sans numérotation.
"""
        response = self.llm.complete(prompt)
        questions_text = str(response)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        questions = [q for q in questions if len(q) > 10 and '?' in q]
        return questions[:3]


class LLMAnswerGenerator(AnswerGenerator):
    """Générateur de réponses utilisant un LLM (Ollama)."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct", temperature: float = 0.1, request_timeout: float = 60.0):
        self.llm = Ollama(model=llm_model, temperature=temperature, request_timeout=request_timeout)
    
    def generate_answer(self, question: str, chunk: Document) -> str:
        prompt = f"""
Tu es un assistant spécialisé en droit administratif. Réponds en français à la question suivante en te basant uniquement sur le texte fourni.

Question: {question}

Texte de référence:
{chunk.text}

Instructions:
1. Réponds uniquement en français
2. Utilise uniquement les informations du texte fourni
3. Sois précis et concis
4. Si la réponse n'est pas dans le texte, dis-le clairement
5. Structure ta réponse de manière claire

Réponse:
"""
        response = self.llm.complete(prompt)
        return str(response).strip()


