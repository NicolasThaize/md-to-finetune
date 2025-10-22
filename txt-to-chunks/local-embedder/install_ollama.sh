#!/bin/bash
# Script d'installation d'Ollama et du modèle Mistral

echo "🚀 Installation d'Ollama et du modèle Mistral"
echo "=============================================="

# Vérifier si Ollama est déjà installé
if command -v ollama &> /dev/null; then
    echo "✅ Ollama déjà installé"
    ollama --version
else
    echo "📥 Installation d'Ollama..."
    
    # Détecter l'OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "🍎 Détection macOS"
        echo "📥 Installation d'Ollama pour macOS..."
        
        # Vérifier si Homebrew est installé
        if command -v brew &> /dev/null; then
            echo "✅ Homebrew détecté, installation via Homebrew"
            brew install ollama
        else
            echo "📥 Installation manuelle d'Ollama"
            echo "1. Téléchargez Ollama depuis: https://ollama.ai/download"
            echo "2. Installez l'application .dmg"
            echo "3. Lancez Ollama depuis Applications"
            echo ""
            echo "⏳ Attendez que Ollama soit installé et lancé, puis appuyez sur Entrée..."
            read -p "Appuyez sur Entrée quand Ollama est installé et lancé..."
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "🐧 Détection Linux"
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "❌ OS non supporté automatiquement"
        echo "📖 Veuillez installer Ollama manuellement: https://ollama.ai"
        exit 1
    fi
fi

echo ""
echo "🤖 Installation du modèle Mistral 7B Instruct..."
ollama pull mistral:7b-instruct

echo ""
echo "🔍 Vérification de l'installation..."
ollama list

echo ""
echo "✅ Installation terminée!"
echo ""
echo "💡 Pour tester l'installation:"
echo "   python test_qa_generation.py"
echo ""
echo "💡 Pour lancer le script principal:"
echo "   python main.py"
