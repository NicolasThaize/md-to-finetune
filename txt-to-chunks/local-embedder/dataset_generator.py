#!/usr/bin/env python3
"""
Générateur de dataset Q/R à partir de documents Markdown.
Utilise des design patterns pour une architecture claire et maintenable.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.ollama import Ollama

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Charger les variables d'environnement si disponibles
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class QAPair:
    """Représente une paire question/réponse."""
    question: str
    answer: str
    source_title: str
    source_level: int
    
    def to_jsonl_format(self) -> Dict[str, Any]:
        """Convertit en format JSONL standard."""
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer}
            ]
        }


class MarkdownParser(ABC):
    """Interface pour le parsing de documents Markdown."""
    
    @abstractmethod
    def parse(self, content: str) -> List[Document]:
        """Parse le contenu Markdown en documents."""
        pass


class HierarchicalMarkdownParser(MarkdownParser):
    """Parser Markdown qui préserve la hiérarchie des titres."""
    
    def parse(self, content: str) -> List[Document]:
        """Parse le contenu en préservant la structure hiérarchique."""
        documents = []
        lines = content.split('\n')
        current_section = {}
        current_content = []
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Sauvegarder la section précédente
                if current_section and current_content:
                    doc = self._create_document_from_section(current_section, current_content)
                    if doc:
                        documents.append(doc)
                
                # Nouvelle section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = {
                    'title': title,
                    'level': level,
                    'line_number': i + 1
                }
                current_content = [line]
            else:
                if current_section:
                    current_content.append(line)
        
        # Dernière section
        if current_section and current_content:
            doc = self._create_document_from_section(current_section, current_content)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _create_document_from_section(self, section: Dict, content: List[str]) -> Optional[Document]:
        """Crée un document à partir d'une section."""
        full_content = '\n'.join(content).strip()
        
        if not full_content or len(full_content) < 10:
            return None
        
        return Document(
            text=full_content,
            metadata={
                'title': section['title'],
                'level': section['level'],
                'line_number': section['line_number'],
                'section_type': 'hierarchy'
            }
        )


class ChunkProcessor(ABC):
    """Interface pour le traitement des chunks."""
    
    @abstractmethod
    def process(self, documents: List[Document]) -> List[Document]:
        """Traite les documents en chunks."""
        pass


class MarkdownChunkProcessor(ChunkProcessor):
    """Processeur de chunks spécialisé pour Markdown."""
    
    def __init__(self):
        self.node_parser = MarkdownNodeParser()
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Découpe les documents en chunks sémantiques."""
        all_chunks = []
        
        for doc in documents:
            try:
                nodes = self.node_parser.get_nodes_from_documents([doc])
                
                for node in nodes:
                    # Enrichir les métadonnées
                    node.metadata.update({
                        'parent_title': doc.metadata.get('title', ''),
                        'parent_level': doc.metadata.get('level', 0),
                        'chunk_type': 'markdown_section'
                    })
                    all_chunks.append(node)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du document {doc.metadata.get('title', 'Sans titre')}: {e}")
                continue
        
        return all_chunks


class QuestionGenerator(ABC):
    """Interface pour la génération de questions."""
    
    @abstractmethod
    def generate_questions(self, chunk: Document) -> List[str]:
        """Génère des questions pour un chunk."""
        pass


class LLMQuestionGenerator(QuestionGenerator):
    """Générateur de questions utilisant un LLM."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct"):
        self.llm = Ollama(
            model=llm_model,
            temperature=0.1,
            request_timeout=60.0
        )
    
    def generate_questions(self, chunk: Document) -> List[str]:
        """Génère des questions pertinentes pour un chunk."""
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
        
        try:
            response = self.llm.complete(prompt)
            questions_text = str(response)
            
            # Parser les questions
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            questions = [q for q in questions if len(q) > 10 and '?' in q]
            
            return questions[:3]  # Limiter à 3 questions max
            
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de questions: {e}")
            return []


class AnswerGenerator(ABC):
    """Interface pour la génération de réponses."""
    
    @abstractmethod
    def generate_answer(self, question: str, chunk: Document) -> str:
        """Génère une réponse pour une question."""
        pass


class LLMAnswerGenerator(AnswerGenerator):
    """Générateur de réponses utilisant un LLM."""
    
    def __init__(self, llm_model: str = "mistral:7b-instruct"):
        self.llm = Ollama(
            model=llm_model,
            temperature=0.1,
            request_timeout=60.0
        )
    
    def generate_answer(self, question: str, chunk: Document) -> str:
        """Génère une réponse basée sur le chunk."""
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
        
        try:
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de réponse: {e}")
            return ""


class QAPairGenerator:
    """Générateur de paires Q/R utilisant le pattern Strategy."""
    
    def __init__(self, question_generator: QuestionGenerator, answer_generator: AnswerGenerator):
        self.question_generator = question_generator
        self.answer_generator = answer_generator
    
    def generate_qa_pairs(self, chunks: List[Document]) -> List[QAPair]:
        """Génère des paires Q/R pour tous les chunks."""
        qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Traitement du chunk {i+1}/{len(chunks)}")
            
            try:
                # Générer des questions
                questions = self.question_generator.generate_questions(chunk)
                
                # Générer des réponses
                for question in questions:
                    answer = self.answer_generator.generate_answer(question, chunk)
                    
                    if question and answer:
                        qa_pair = QAPair(
                            question=question,
                            answer=answer,
                            source_title=chunk.metadata.get('title', 'Sans titre'),
                            source_level=chunk.metadata.get('level', 0)
                        )
                        qa_pairs.append(qa_pair)
                        logger.info(f"✅ Paire Q/R générée: {question[:50]}...")
                
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du chunk {i+1}: {e}")
                continue
        
        return qa_pairs


class DatasetExporter(ABC):
    """Interface pour l'export de datasets."""
    
    @abstractmethod
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte les paires Q/R dans un fichier."""
        pass


class JSONLExporter(DatasetExporter):
    """Exporteur au format JSONL."""
    
    def export(self, qa_pairs: List[QAPair], output_path: Path) -> None:
        """Exporte en format JSONL standard."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                json.dump(qa_pair.to_jsonl_format(), f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"✅ {len(qa_pairs)} paires Q/R exportées vers {output_path}")


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
    
    def generate_dataset(self, 
                        markdown_file: Path, 
                        output_file: Path) -> List[QAPair]:
        """Génère un dataset complet à partir d'un fichier Markdown."""
        logger.info(f"🚀 Génération du dataset depuis {markdown_file}")
        
        # 1. Charger et parser le Markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = self.markdown_parser.parse(content)
        logger.info(f"📄 {len(documents)} sections parsées")
        
        # 2. Traiter en chunks
        chunks = self.chunk_processor.process(documents)
        logger.info(f"📝 {len(chunks)} chunks créés")
        
        # 3. Générer les paires Q/R
        qa_pairs = self.qa_generator.generate_qa_pairs(chunks)
        logger.info(f"❓ {len(qa_pairs)} paires Q/R générées")
        
        # 4. Exporter
        self.exporter.export(qa_pairs, output_file)
        
        return qa_pairs


class DatasetGeneratorFactory:
    """Factory pour créer des générateurs de dataset."""
    
    @staticmethod
    def create_default_generator(llm_model: str = "mistral:7b-instruct") -> DatasetGenerator:
        """Crée un générateur avec la configuration par défaut."""
        # Composants
        markdown_parser = HierarchicalMarkdownParser()
        chunk_processor = MarkdownChunkProcessor()
        
        # LLM components
        question_generator = LLMQuestionGenerator(llm_model)
        answer_generator = LLMAnswerGenerator(llm_model)
        qa_generator = QAPairGenerator(question_generator, answer_generator)
        
        # Export
        exporter = JSONLExporter()
        
        return DatasetGenerator(
            markdown_parser=markdown_parser,
            chunk_processor=chunk_processor,
            qa_generator=qa_generator,
            exporter=exporter
        )


def main():
    """Fonction principale."""
    # Configuration
    markdown_file = Path("sources/droitadminSmall.md")
    output_file = Path("training_data.jsonl")
    
    try:
        # Créer le générateur
        generator = DatasetGeneratorFactory.create_default_generator()
        
        # Générer le dataset
        qa_pairs = generator.generate_dataset(markdown_file, output_file)
        
        print(f"\n✅ Dataset généré avec succès!")
        print(f"📊 {len(qa_pairs)} paires Q/R créées")
        print(f"💾 Fichier: {output_file}")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
