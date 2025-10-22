#!/usr/bin/env python3
"""
Script de test pour vérifier le fonctionnement du processeur avec Mistral AI.
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

def test_imports():
    """Test des imports nécessaires."""
    print("🔍 Test des imports...")
    
    try:
        from main_mistral import MarkdownDocumentProcessorMistral
        print("✅ Import de MarkdownDocumentProcessorMistral réussi")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    
    return True

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

def test_mistral_api_key():
    """Test de la présence de la clé API Mistral AI."""
    print("\n🔑 Test de la clé API Mistral AI...")
    
    # Vérifier si le fichier .env existe
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ Fichier .env trouvé: {env_file}")
    else:
        print("⚠️  Fichier .env non trouvé")
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        print("✅ Clé API Mistral AI trouvée")
        print(f"   Longueur: {len(api_key)} caractères")
        print(f"   Début: {api_key[:10]}...")
        return True
    else:
        print("❌ Clé API Mistral AI non trouvée")
        print("   Vérifiez que votre fichier .env contient: MISTRAL_API_KEY=votre_cle")
        return False

def test_processor_initialization():
    """Test de l'initialisation du processeur."""
    print("\n🏗️ Test de l'initialisation du processeur...")
    
    try:
        from main_mistral import MarkdownDocumentProcessorMistral
        
        processor = MarkdownDocumentProcessorMistral(
            markdown_file_path="sources/droitadminSmall.md",
            storage_dir="./test_storage_mistral"
        )
        print("✅ Processeur Mistral AI initialisé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def test_markdown_parsing():
    """Test du parsing du Markdown."""
    print("\n📄 Test du parsing Markdown...")
    
    try:
        from main_mistral import MarkdownDocumentProcessorMistral
        
        processor = MarkdownDocumentProcessorMistral(
            markdown_file_path="sources/droitadminSmall.md",
            storage_dir="./test_storage_mistral"
        )
        
        documents = processor.load_and_parse_markdown()
        print(f"✅ Parsing réussi: {len(documents)} sections trouvées")
        
        # Afficher quelques exemples
        for i, doc in enumerate(documents[:3]):
            title = doc.metadata.get('title', 'Sans titre')
            level = doc.metadata.get('level', 0)
            print(f"   Section {i+1}: {title} (niveau {level})")
        
        return True
    except Exception as e:
        print(f"❌ Erreur de parsing: {e}")
        return False

def test_mistral_embeddings():
    """Test des embeddings Mistral AI."""
    print("\n🧠 Test des embeddings Mistral AI...")
    
    try:
        from llama_index.embeddings.mistralai import MistralAIEmbedding
        
        # Tester l'initialisation des embeddings
        embed_model = MistralAIEmbedding(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        
        # Test simple d'embedding
        test_text = "Test d'embedding avec Mistral AI"
        embedding = embed_model.get_text_embedding(test_text)
        
        print(f"✅ Embeddings Mistral AI fonctionnels")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Premières valeurs: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur avec les embeddings Mistral AI: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 TESTS DU PROCESSEUR MISTRAL AI")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Fichier source", test_file_existence),
        ("Clé API Mistral", test_mistral_api_key),
        ("Initialisation", test_processor_initialization),
        ("Parsing Markdown", test_markdown_parsing),
        ("Embeddings Mistral", test_mistral_embeddings)
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
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! Le script Mistral AI est prêt à être utilisé.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
