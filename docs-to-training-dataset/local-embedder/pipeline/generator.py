import logging
from pathlib import Path
from typing import List

from llama_index.core import Document

from core.parsing import MarkdownParser
from core.chunking import ChunkProcessor
from core.domain import QAPair
from core.qa import QuestionGenerator, AnswerGenerator
from exporters.base import DatasetExporter


logger = logging.getLogger(__name__)


class QAPairGenerator:
    """Générateur de paires Q/R utilisant le pattern Strategy."""
    
    def __init__(self, question_generator: QuestionGenerator, answer_generator: AnswerGenerator):
        self.question_generator = question_generator
        self.answer_generator = answer_generator
    
    def generate_qa_pairs(self, chunks: List[Document]) -> List[QAPair]:
        qa_pairs: List[QAPair] = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Traitement du chunk {i+1}/{len(chunks)}")
            questions = self.question_generator.generate_questions(chunk)
            for question in questions:
                answer = self.answer_generator.generate_answer(question, chunk)
                if question and answer:
                    qa_pairs.append(QAPair(
                        question=question,
                        answer=answer,
                        source_title=chunk.metadata.get('title', 'Sans titre'),
                        source_level=chunk.metadata.get('level', 0)
                    ))
        return qa_pairs


class DatasetGenerator:
    """Générateur principal utilisant le pattern Facade."""
    
    def __init__(self, 
                 markdown_parser: MarkdownParser,
                 chunk_processor: ChunkProcessor,
                 qa_generator: QAPairGenerator,
                 exporter: DatasetExporter):
        self.markdown_parser = markdown_parser
        self.chunk_processor = chunk_processor
        self.qa_generator = qa_generator
        self.exporter = exporter
    
    def generate_dataset(self, markdown_file: Path, output_file: Path) -> List[QAPair]:
        logger.info(f"🚀 Génération du dataset depuis {markdown_file}")
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        documents = self.markdown_parser.parse(content)
        logger.info(f"📄 {len(documents)} sections parsées")
        chunks = self.chunk_processor.process(documents)
        logger.info(f"📝 {len(chunks)} chunks créés")
        qa_pairs = self.qa_generator.generate_qa_pairs(chunks)
        logger.info(f"❓ {len(qa_pairs)} paires Q/R générées")
        self.exporter.export(qa_pairs, output_file)
        return qa_pairs


