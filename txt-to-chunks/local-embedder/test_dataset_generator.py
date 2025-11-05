#!/usr/bin/env python3
"""
Tests unitaires/intégration du générateur de dataset.
Objectif: valider les briques (parser markdown hiérarchique, chunking), les générateurs
Q/R (mockés), l'assemblage (QAPairGenerator, Factory) et les exporteurs:
- messages JSONL, question/answer JSONL, user/assistant JSONL
- format Mistral CSV via tokenizer.apply_chat_template (mocké)
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Charger les variables d'environnement
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Fichier .env chargé")
except ImportError:
    print("⚠️  python-dotenv non installé")

# Ajouter le répertoire courant au path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test des imports nécessaires."""
    print("🔍 Test des imports...")
    
    try:
        from core.domain import QAPair
        from core.parsing import MarkdownParser, HierarchicalMarkdownParser
        from core.chunking import ChunkProcessor, MarkdownChunkProcessor
        from core.qa import QuestionGenerator, AnswerGenerator
        from llm.ollama_generators import LLMQuestionGenerator, LLMAnswerGenerator
        from pipeline.generator import QAPairGenerator, DatasetGenerator
        from pipeline.factory import DatasetGeneratorFactory
        from exporters.base import DatasetExporter
        from exporters import (
            MessagesFormatExporter,
            QuestionAnswerFormatExporter,
            UserAssistantFormatExporter,
            MistralChatTemplateExporter,
        )
        print("✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_qa_pair_dataclass():
    """Test de la classe QAPair."""
    print("\n📝 Test de la classe QAPair...")
    
    try:
        from core.domain import QAPair
        
        # Créer une paire Q/R
        qa_pair = QAPair(
            question="Qu'est-ce que l'administration?",
            answer="L'administration est l'ensemble des personnes publiques françaises.",
            source_title="Introduction",
            source_level=1
        )
        
        # Test des propriétés
        assert qa_pair.question == "Qu'est-ce que l'administration?"
        assert qa_pair.answer == "L'administration est l'ensemble des personnes publiques françaises."
        assert qa_pair.source_title == "Introduction"
        assert qa_pair.source_level == 1
        
        print("✅ QAPair fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec QAPair: {e}")
        return False

def test_markdown_parser():
    """Test du parser Markdown."""
    print("\n📄 Test du parser Markdown...")
    
    try:
        from core.parsing import HierarchicalMarkdownParser
        
        parser = HierarchicalMarkdownParser()
        
        # Test avec un contenu Markdown simple
        test_content = """# Titre principal

Contenu du titre principal.

## Sous-titre

Contenu du sous-titre.

### Sous-sous-titre

Contenu du sous-sous-titre.
"""
        
        documents = parser.parse(test_content)
        
        # Vérifications
        assert len(documents) == 3, f"Attendu 3 documents, obtenu {len(documents)}"
        
        # Vérifier le premier document
        first_doc = documents[0]
        assert first_doc.metadata['title'] == "Titre principal"
        assert first_doc.metadata['level'] == 1
        assert "Contenu du titre principal" in first_doc.text
        
        print("✅ Parser Markdown fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec le parser Markdown: {e}")
        return False

def test_chunk_processor():
    """Test du processeur de chunks."""
    print("\n📝 Test du processeur de chunks...")
    
    try:
        from core.chunking import MarkdownChunkProcessor
        from llama_index.core import Document
        
        processor = MarkdownChunkProcessor()
        
        # Créer des documents de test
        documents = [
            Document(
                text="Contenu du document 1",
                metadata={'title': 'Document 1', 'level': 1}
            ),
            Document(
                text="Contenu du document 2",
                metadata={'title': 'Document 2', 'level': 2}
            )
        ]
        
        chunks = processor.process(documents)
        
        # Vérifications
        assert len(chunks) > 0, "Aucun chunk généré"
        
        # Vérifier les métadonnées enrichies
        for chunk in chunks:
            assert 'parent_title' in chunk.metadata
            assert 'parent_level' in chunk.metadata
            assert 'chunk_type' in chunk.metadata
        
        print("✅ Processeur de chunks fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec le processeur de chunks: {e}")
        return False

def test_llm_components():
    """Test des composants LLM."""
    print("\n🤖 Test des composants LLM...")
    
    try:
        from llm.ollama_generators import LLMQuestionGenerator, LLMAnswerGenerator
        from llama_index.core import Document
        
        # Test avec mock pour éviter les appels réels
        with patch('llm.ollama_generators.Ollama') as mock_ollama:
            # Configurer le mock
            mock_llm = Mock()
            mock_llm.complete.return_value = Mock()
            mock_llm.complete.return_value.__str__ = Mock(return_value="Question test?\nAutre question?")
            mock_ollama.return_value = mock_llm
            
            # Test du générateur de questions
            question_gen = LLMQuestionGenerator("test-model")
            doc = Document(text="Test content", metadata={'title': 'Test'})
            questions = question_gen.generate_questions(doc)
            
            assert isinstance(questions, list)
            
            # Test du générateur de réponses
            mock_llm.complete.return_value.__str__ = Mock(return_value="Réponse test")
            answer_gen = LLMAnswerGenerator("test-model")
            answer = answer_gen.generate_answer("Question test?", doc)
            
            assert isinstance(answer, str)
        
        print("✅ Composants LLM fonctionnent correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec les composants LLM: {e}")
        return False

def test_qa_pair_generator():
    """Test du générateur de paires Q/R."""
    print("\n❓ Test du générateur de paires Q/R...")
    
    try:
        from pipeline.generator import QAPairGenerator
        from core.qa import QuestionGenerator, AnswerGenerator
        from llama_index.core import Document
        
        # Mock des générateurs
        class MockQuestionGenerator(QuestionGenerator):
            def generate_questions(self, chunk):
                return ["Question test 1?", "Question test 2?"]
        
        class MockAnswerGenerator(AnswerGenerator):
            def generate_answer(self, question, chunk):
                return f"Réponse pour: {question}"
        
        # Test du générateur
        qa_gen = QAPairGenerator(
            MockQuestionGenerator(),
            MockAnswerGenerator()
        )
        
        # Créer un chunk de test
        chunk = Document(
            text="Contenu de test",
            metadata={'title': 'Test', 'level': 1}
        )
        
        qa_pairs = qa_gen.generate_qa_pairs([chunk])
        
        # Vérifications
        assert len(qa_pairs) == 2, f"Attendu 2 paires, obtenu {len(qa_pairs)}"
        assert qa_pairs[0].question == "Question test 1?"
        assert qa_pairs[0].answer == "Réponse pour: Question test 1?"
        
        print("✅ Générateur de paires Q/R fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec le générateur de paires Q/R: {e}")
        return False

def test_messages_exporter():
    """Test de l'exporteur Messages (JSONL)."""
    print("\n📤 Test de l'exporteur Messages (JSONL)...")
    
    try:
        from core.domain import QAPair
        from exporters import MessagesFormatExporter
        import tempfile
        
        exporter = MessagesFormatExporter()
        
        # Créer des paires Q/R de test
        qa_pairs = [
            QAPair(
                question="Question 1?",
                answer="Réponse 1",
                source_title="Source 1",
                source_level=1
            ),
            QAPair(
                question="Question 2?",
                answer="Réponse 2",
                source_title="Source 2",
                source_level=2
            )
        ]
        
        # Test d'export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = Path(f.name)
        
        exporter.export(qa_pairs, temp_path)
        
        # Vérifier le contenu
        with open(temp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 2, f"Attendu 2 lignes, obtenu {len(lines)}"
        
        # Vérifier le format JSON
        for line in lines:
            data = json.loads(line.strip())
            assert 'messages' in data
            assert len(data['messages']) == 2
            assert data['messages'][0]['role'] == 'user'
            assert data['messages'][1]['role'] == 'assistant'
        
        # Nettoyer
        temp_path.unlink()
        
        print("✅ Exporteur Messages fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec l'exporteur Messages: {e}")
        return False

def test_mistral_exporter():
    """Test de l'exporteur Mistral (CSV) avec mock du tokenizer."""
    print("\n🟦 Test de l'exporteur Mistral (CSV)...")
    
    try:
        from core.domain import QAPair
        from exporters.mistral_template import MistralChatTemplateExporter
        import tempfile
        
        with patch('exporters.mistral_template.AutoTokenizer') as mock_tok:
            # Mock tokenizer behavior
            mock_instance = Mock()
            mock_instance.apply_chat_template.return_value = "<s>[INST] Q [/INST] A</s>"
            mock_tok.from_pretrained.return_value = mock_instance
            
            exporter = MistralChatTemplateExporter(model_name="mock-model")
            
            qa_pairs = [
                QAPair(question="Q1?", answer="A1", source_title="S1", source_level=1),
                QAPair(question="Q2?", answer="A2", source_title="S2", source_level=2),
            ]
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
                temp_path = Path(f.name)
            
            exporter.export(qa_pairs, temp_path)
            
            # Vérifier le contenu CSV
            import csv
            with open(temp_path, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
            
            # Header + 2 rows
            assert len(reader) == 3, f"Attendu 3 lignes (1 header + 2), obtenu {len(reader)}"
            assert reader[0] == ['formatted_text']
            assert reader[1][0].startswith('<s>[INST]')
            
            temp_path.unlink()
        
        print("✅ Exporteur Mistral fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec l'exporteur Mistral: {e}")
        return False

def test_factory():
    """Test de la factory."""
    print("\n🏭 Test de la factory...")
    
    try:
        from pipeline.factory import DatasetGeneratorFactory
        
        # Test de création avec mock
        with patch('llm.ollama_generators.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            
            generator = DatasetGeneratorFactory.create_default_generator("test-model")
            
            # Vérifications
            assert generator is not None
            assert generator.markdown_parser is not None
            assert generator.chunk_processor is not None
            assert generator.qa_generator is not None
            assert generator.exporter is not None
        
        print("✅ Factory fonctionne correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur avec la factory: {e}")
        return False

def test_file_existence():
    """Test de l'existence du fichier source."""
    print("\n📁 Test de l'existence du fichier source...")
    
    markdown_file = Path("sources/droitadminSmall.md")
    if markdown_file.exists():
        print(f"✅ Fichier trouvé: {markdown_file}")
        print(f"   Taille: {markdown_file.stat().st_size} bytes")
        return True
    else:
        print(f"❌ Fichier non trouvé: {markdown_file}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 TESTS DU GÉNÉRATEUR DE DATASET")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("QAPair dataclass", test_qa_pair_dataclass),
        ("Parser Markdown", test_markdown_parser),
        ("Processeur de chunks", test_chunk_processor),
        ("Composants LLM", test_llm_components),
        ("Générateur Q/R", test_qa_pair_generator),
        ("Exporteur Messages", test_messages_exporter),
        ("Exporteur Mistral", test_mistral_exporter),
        ("Factory", test_factory),
        ("Fichier source", test_file_existence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des tests
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! Le générateur de dataset est prêt.")
        print("\n💡 Pour générer un dataset:")
        print("   python main.py --format messages --input sources/droitadminSmall.md --output training_data.jsonl")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
