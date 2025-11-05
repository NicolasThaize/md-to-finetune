#!/usr/bin/env python3
"""
Tests d'intégration légère du processeur local.
Objectif: valider l'environnement et la chaîne locale (import, fichier source,
parsing Markdown, initialisation des embeddings HF, disponibilité GPU, téléchargement
du modèle). Ces tests n'impliquent ni exporteurs ni Ollama.
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
        from main import MarkdownDocumentProcessorLocal
        print("✅ Import de MarkdownDocumentProcessorLocal réussi")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    
    return True

def test_file_existence():
    """Test de l'existence du fichier source."""
    print("\n📁 Test de l'existence du fichier source...")
    
    markdown_file = Path("./sources/droitadminSmall.md")
    if markdown_file.exists():
        print(f"✅ Fichier trouvé: {markdown_file}")
        print(f"   Taille: {markdown_file.stat().st_size} bytes")
        return True
    else:
        print(f"❌ Fichier non trouvé: {markdown_file}")
        return False

def test_processor_initialization():
    """Test de l'initialisation du processeur."""
    print("\n🏗️ Test de l'initialisation du processeur...")
    
    try:
        from main import MarkdownDocumentProcessorLocal
        
        processor = MarkdownDocumentProcessorLocal(
            markdown_file_path="../sources/droitadminSmall.md",
            storage_dir="./test_storage_local"
        )
        print("✅ Processeur local initialisé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def test_markdown_parsing():
    """Test du parsing du Markdown."""
    print("\n📄 Test du parsing Markdown...")
    
    try:
        from main import MarkdownDocumentProcessorLocal
        
        processor = MarkdownDocumentProcessorLocal(
            markdown_file_path="./sources/droitadminSmall.md",
            storage_dir="./test_storage_local"
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

def test_local_embeddings():
    """Test des embeddings locaux."""
    print("\n🧠 Test des embeddings locaux...")
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Tester l'initialisation des embeddings
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/distiluse-base-multilingual-cased",
            device="cpu"  # Utiliser CPU pour le test
        )
        
        # Test simple d'embedding
        test_text = "Test d'embedding avec modèle local"
        embedding = embed_model.get_text_embedding(test_text)
        
        print(f"✅ Embeddings locaux fonctionnels")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Premières valeurs: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur avec les embeddings locaux: {e}")
        return False

def test_gpu_availability():
    """Test de la disponibilité du GPU."""
    print("\n🖥️ Test de la disponibilité du GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  Aucun GPU détecté, utilisation du CPU")
            return True  # Ce n'est pas une erreur, juste un avertissement
    except ImportError:
        print("⚠️  PyTorch non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification GPU: {e}")
        return False

def test_model_download():
    """Test du téléchargement du modèle."""
    print("\n📥 Test du téléchargement du modèle...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Tester le téléchargement du modèle
        model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")
        
        # Test simple
        test_texts = ["Test en français", "Test in English"]
        embeddings = model.encode(test_texts)
        
        print(f"✅ Modèle téléchargé et fonctionnel")
        print(f"   Dimension des embeddings: {embeddings.shape[1]}")
        print(f"   Nombre de textes testés: {len(test_texts)}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du modèle: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 TESTS DU PROCESSEUR LOCAL (DISTILUSE)")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Fichier source", test_file_existence),
        ("Initialisation", test_processor_initialization),
        ("Parsing Markdown", test_markdown_parsing),
        ("Embeddings locaux", test_local_embeddings),
        ("Disponibilité GPU", test_gpu_availability),
        ("Téléchargement modèle", test_model_download)
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
        print("🎉 Tous les tests sont passés! Le script local est prêt à être utilisé.")
        print("\n💡 Avantages de la version locale:")
        print("   ✅ Gratuit (pas de coûts d'API)")
        print("   ✅ Sans limites (pas de rate limits)")
        print("   ✅ Rapide (pas de latence réseau)")
        print("   ✅ Privé (données restent locales)")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
