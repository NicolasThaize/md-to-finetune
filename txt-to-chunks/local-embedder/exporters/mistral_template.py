"""Mistral chat template format exporter."""

import csv
import logging
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from exporters.base import DatasetExporter
from dataset_generator import QAPair

logger = logging.getLogger(__name__)


class MistralChatTemplateExporter(DatasetExporter):
    """Exporteur au format Mistral Chat Template (CSV)."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialise l'exporteur Mistral.
        
        Args:
            model_name: Nom du modèle Mistral à utiliser pour le tokenizer
        """
        logger.info(f"Chargement du tokenizer Mistral: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("✅ Tokenizer chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du tokenizer: {e}")
            raise ValueError(f"Impossible de charger le tokenizer pour {model_name}. "
                           f"Assurez-vous que le modèle est disponible.")
    
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte en format CSV avec formatted_text."""
        # Ensure .csv extension
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        
        formatted_texts = []
        
        for qa_pair in qa_pairs:
            # Convert QAPair to messages format
            messages = [
                {"role": "user", "content": qa_pair.question},
                {"role": "assistant", "content": qa_pair.answer}
            ]
            
            # Apply chat template
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
            except Exception as e:
                logger.warning(f"Erreur lors de l'application du template pour une paire Q/R: {e}")
                continue
        
        # Write to CSV file
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['formatted_text'])
            # Write each formatted conversation
            for formatted_text in formatted_texts:
                writer.writerow([formatted_text])
        
        logger.info(f"✅ {len(formatted_texts)} paires Q/R exportées vers {output_path}")

