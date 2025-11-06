"""Base exporter interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.domain import QAPair


class DatasetExporter(ABC):
    """Interface pour l'export de datasets."""
    
    @abstractmethod
    def export(self, qa_pairs: List['QAPair'], output_path: Path) -> None:
        """Exporte les paires Q/R dans un fichier."""
        pass

