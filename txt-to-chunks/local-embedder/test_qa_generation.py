#!/usr/bin/env python3
"""
Tests d'intégration orientés LLM avec Ollama pour la génération Q/R.
Objectif: vérifier client/serveur Ollama, une génération simple, puis la
génération de questions/réponses à partir d'un vrai chunk Markdown via
MarkdownDocumentProcessorLocal. Dépend de l'environnement local (Ollama + modèle
mistral).
"""

import os
import sys
from pathlib import Path

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # Charge le fichier .env
    print("✅ Fichier .env chargé")
except ImportError:
    print("⚠️  python-dotenv non installé, utilisation des variables d'environnement système")

# Ajouter le répertoire courant au path
sys.path.append(str(Path(__file__).parent))

def test_ollama_installation():
    """Test de l'installation d'Ollama."""
    print("🔍 Test de l'installation d'Ollama...")
    
    try:
        import ollama
        print("✅ Ollama Python client installé")
        return True
    except ImportError:
        print("❌ Ollama Python client non installé")
        print("   Installez avec: pip install ollama")
        return False

def test_ollama_server():
    """Test de la connexion au serveur Ollama."""
    print("\n🖥️ Test de la connexion au serveur Ollama...")
    
    try:
        import ollama
        
        # Tester la connexion
        models = ollama.list()
        print("✅ Serveur Ollama accessible")
        print(f"   Modèles disponibles: {models['models']}")
        
        # Vérifier si Mistral est installé
        mistral_models = [m for m in models['models'] if 'mistral' in m['model'].lower()]
        if mistral_models:
            print(f"✅ Modèle Mistral trouvé: {mistral_models[0]['model']}")
            return True
        else:
            print("⚠️  Modèle Mistral non trouvé")
            print("   Installez avec: ollama pull mistral:7b-instruct")
            return False
            
    except Exception as e:
        print(f"❌ Erreur de connexion à Ollama: {e}")
        print("   Assurez-vous qu'Ollama est installé et démarré:")
        print("   1. Téléchargez depuis: https://ollama.ai")
        print("   2. Installez le modèle: ollama pull mistral:7b-instruct")
        return False

def test_llm_generation():
    """Test de la génération avec le LLM."""
    print("\n🤖 Test de la génération avec le LLM...")
    
    try:
        from llama_index.llms.ollama import Ollama
        
        # Initialiser le LLM
        llm = Ollama(
            model="mistral:7b-instruct",
            temperature=0.1,
            request_timeout=30.0
        )
        
        # Test simple
        test_prompt = "Bonjour, comment allez-vous?"
        response = llm.complete(test_prompt)
        
        print("✅ LLM fonctionnel")
        print(f"   Réponse: {str(response)[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Erreur avec le LLM: {e}")
        return False

def test_qa_generation():
    """Test de la génération de paires Q/R."""
    print("\n❓ Test de la génération de paires Q/R...")
    
    try:
        from main import MarkdownDocumentProcessorLocal
        
        processor = MarkdownDocumentProcessorLocal(
            markdown_file_path="./sources/droitadminSmall.md",
            storage_dir="./test_storage_qa",
            llm_model="mistral:7b-instruct"
        )
        
        # Charger un petit échantillon
        documents = processor.load_and_parse_markdown()
        if not documents:
            print("❌ Aucun document trouvé")
            return False
        
        # Prendre seulement le premier document pour le test
        test_doc = documents[0]
        print(f"✅ Document de test: {test_doc.metadata.get('title', 'Sans titre')}")
        
        # Test de génération de questions
        questions = processor._generate_questions_for_chunk(test_doc)
        if questions:
            print(f"✅ Questions générées: {len(questions)}")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q}")
        else:
            print("❌ Aucune question générée")
            return False
        
        # Test de génération de réponse
        if questions:
            answer = processor._generate_answer_for_question(questions[0], test_doc)
            if answer:
                print(f"✅ Réponse générée: {answer[:100]}...")
                return True
            else:
                print("❌ Aucune réponse générée")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test Q/R: {e}")
        return False

def test_jsonl_format():
    """Test du format JSONL."""
    print("\n📄 Test du format JSONL...")
    
    try:
        import json
        
        # Créer un exemple de paire Q/R
        qa_pair = {
            "messages": [
                {"role": "user", "content": "Qu'est-ce que l'administration?"},
                {"role": "assistant", "content": "L'administration est l'ensemble des personnes publiques françaises."}
            ]
        }
        
        # Tester la sérialisation JSON
        json_str = json.dumps(qa_pair, ensure_ascii=False)
        parsed = json.loads(json_str)
        
        print("✅ Format JSONL valide")
        print(f"   Exemple: {json_str[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Erreur avec le format JSONL: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 TESTS DE GÉNÉRATION Q/R")
    print("=" * 60)
    
    tests = [
        ("Installation Ollama", test_ollama_installation),
        ("Serveur Ollama", test_ollama_server),
        ("Génération LLM", test_llm_generation),
        ("Génération Q/R", test_qa_generation),
        ("Format JSONL", test_jsonl_format)
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
        print("🎉 Tous les tests sont passés! La génération Q/R est prête.")
        print("\n💡 Pour générer des données d'entraînement:")
        print("   python main.py")
        print("   Choisissez l'option 2")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        print("\n🔧 Solutions possibles:")
        print("   1. Installez Ollama: https://ollama.ai")
        print("   2. Installez le modèle: ollama pull mistral:7b-instruct")
        print("   3. Vérifiez que le serveur Ollama fonctionne")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
