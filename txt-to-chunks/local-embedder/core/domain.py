from dataclasses import dataclass


@dataclass
class QAPair:
    """Représente une paire question/réponse."""
    question: str
    answer: str
    source_title: str
    source_level: int
