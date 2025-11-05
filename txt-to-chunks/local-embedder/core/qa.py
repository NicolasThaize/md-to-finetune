from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document


class QuestionGenerator(ABC):
    """Interface pour la génération de questions."""
    
    @abstractmethod
    def generate_questions(self, chunk: Document) -> List[str]:
        """Génère des questions pour un chunk."""
        raise NotImplementedError


class AnswerGenerator(ABC):
    """Interface pour la génération de réponses."""
    
    @abstractmethod
    def generate_answer(self, question: str, chunk: Document) -> str:
        """Génère une réponse pour une question."""
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document


class QuestionGenerator(ABC):
    """Interface pour la génération de questions."""
    
    @abstractmethod
    def generate_questions(self, chunk: Document) -> List[str]:
        """Génère des questions pour un chunk."""
        pass


class AnswerGenerator(ABC):
    """Interface pour la génération de réponses."""
    
    @abstractmethod
    def generate_answer(self, question: str, chunk: Document) -> str:
        """Génère une réponse pour une question."""
        pass


