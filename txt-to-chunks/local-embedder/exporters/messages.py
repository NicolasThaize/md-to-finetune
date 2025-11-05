"""Messages format exporter (ChatML/Harmony format)."""

import json
import logging
from pathlib import Path
from typing import List

from exporters.base import DatasetExporter
from dataset_generator import QAPair

logger = logging.getLogger(__name__)


class MessagesFormatExporter(DatasetExporter):
    """Exporteur au format Messages (ChatML/Harmony)."""
    
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte en format JSONL avec messages."""
        # Ensure .jsonl extension
        if output_path.suffix != '.jsonl':
            output_path = output_path.with_suffix('.jsonl')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                json_obj = {
                    "messages": [
                        {"role": "user", "content": qa_pair.question},
                        {"role": "assistant", "content": qa_pair.answer}
                    ]
                }
                json.dump(json_obj, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"✅ {len(qa_pairs)} paires Q/R exportées vers {output_path}")

