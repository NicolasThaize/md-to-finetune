#!/bin/bash
# Script d'installation spécifique pour macOS

echo "🍎 Installation Ollama sur macOS"
echo "================================="

# Vérifier si Ollama est déjà installé
if command -v ollama &> /dev/null; then
    echo "✅ Ollama déjà installé"
    ollama --version
else
    echo "📥 Installation d'Ollama..."
    
    # Vérifier si Homebrew est installé
    if command -v brew &> /dev/null; then
        echo "✅ Homebrew détecté"
        echo "📦 Installation d'Ollama via Homebrew..."
        brew install ollama
        
        # Démarrer Ollama
        echo "🚀 Démarrage d'Ollama..."
        brew services start ollama
    else
        echo "📥 Installation manuelle d'Ollama"
        echo ""
        echo "🔗 Téléchargez Ollama depuis: https://ollama.ai/download"
        echo "📱 Installez l'application .dmg"
        echo "🚀 Lancez Ollama depuis Applications"
        echo ""
        echo "⏳ Attendez que Ollama soit installé et lancé..."
        read -p "Appuyez sur Entrée quand Ollama est installé et lancé..."
    fi
fi

echo ""
echo "🤖 Installation du modèle Mistral 7B Instruct..."
echo "⏳ Cela peut prendre 5-10 minutes (4.1GB à télécharger)..."

# Attendre que Ollama soit prêt
echo "🔄 Vérification que Ollama est prêt..."
sleep 5

# Essayer plusieurs fois si nécessaire
for i in {1..5}; do
    if ollama list &> /dev/null; then
        echo "✅ Ollama est prêt"
        break
    else
        echo "⏳ Attente d'Ollama... (tentative $i/5)"
        sleep 10
    fi
done

# Installer le modèle
echo "📥 Téléchargement du modèle Mistral 7B Instruct..."
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
