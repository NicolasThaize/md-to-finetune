"""Mistral fine-tuning JSONL exporter."""

import json
import logging
from pathlib import Path
from typing import List

from exporters.base import DatasetExporter
from core.domain import QAPair

logger = logging.getLogger(__name__)


class MistralChatTemplateExporter(DatasetExporter):
    """Exporteur JSONL pour l'affinage de Mistral Instruct.

    Produit un fichier .jsonl où chaque ligne contient un objet
    {"text": "<s>[INST] instruction [/INST] response </s>"}.
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        """Initialise l'exporteur Mistral (sans dépendre d'un tokenizer)."""
        self.model_name = model_name
        logger.info(f"Exporter initialisé pour le modèle: {model_name}")
    
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte en JSONL pour l'affinage de Mistral Instruct.

        Chaque ligne contient un objet JSON {"text": "<s>[INST] question [/INST] answer </s>"}.
        """
        # Ensure .jsonl extension
        if output_path.suffix != '.jsonl':
            output_path = output_path.with_suffix('.jsonl')

        num_written = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                try:
                    text = f"<s>[INST] {qa_pair.question} [/INST] {qa_pair.answer} </s>"
                    line_obj = {"text": text}
                    f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                    num_written += 1
                except Exception as e:
                    logger.warning(
                        f"Erreur lors de la sérialisation d'une paire Q/R: {e}")
                    continue

        logger.info(f"✅ {num_written} paires Q/R exportées vers {output_path}")

