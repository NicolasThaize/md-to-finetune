"""Question/Answer format exporter."""

import json
import logging
from pathlib import Path
from typing import List

from exporters.base import DatasetExporter
from core.domain import QAPair

logger = logging.getLogger(__name__)


class QuestionAnswerFormatExporter(DatasetExporter):
    """Exporteur au format Question/Answer."""
    
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte en format JSONL avec question/answer."""
        # Ensure .jsonl extension
        if output_path.suffix != '.jsonl':
            output_path = output_path.with_suffix('.jsonl')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                json_obj = {
                    "question": qa_pair.question,
                    "answer": qa_pair.answer
                }
                json.dump(json_obj, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"✅ {len(qa_pairs)} paires Q/R exportées vers {output_path}")

