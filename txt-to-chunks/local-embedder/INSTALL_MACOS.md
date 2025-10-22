# Installation sur macOS

## 🍎 Guide d'installation Ollama sur macOS

### **Option 1 : Installation automatique (recommandée)**

```bash
# Utiliser le script d'installation macOS
./install_macos.sh
```

### **Option 2 : Installation via Homebrew**

```bash
# Installer Homebrew si pas déjà fait
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer Ollama
brew install ollama

# Démarrer Ollama
brew services start ollama

# Installer le modèle Mistral
ollama pull mistral:7b-instruct
```

### **Option 3 : Installation manuelle**

1. **Télécharger Ollama**
   - Allez sur [https://ollama.ai/download](https://ollama.ai/download)
   - Téléchargez le fichier `.dmg` pour macOS

2. **Installer Ollama**
   - Ouvrez le fichier `.dmg` téléchargé
   - Glissez l'application Ollama dans le dossier Applications
   - Lancez Ollama depuis Applications

3. **Installer le modèle Mistral**
   ```bash
   ollama pull mistral:7b-instruct
   ```

## 🔧 **Vérification de l'installation**

```bash
# Vérifier qu'Ollama fonctionne
ollama list

# Tester le modèle
ollama run mistral:7b-instruct
```

## 🚨 **Dépannage**

### **Erreur "command not found: ollama"**
- Assurez-vous qu'Ollama est installé et lancé
- Redémarrez votre terminal
- Vérifiez que Ollama est dans votre PATH

### **Erreur de connexion**
- Vérifiez qu'Ollama est en cours d'exécution
- Redémarrez Ollama depuis Applications
- Vérifiez qu'aucun firewall ne bloque Ollama

### **Modèle ne se télécharge pas**
- Vérifiez votre connexion internet
- Assurez-vous d'avoir assez d'espace disque (5GB minimum)
- Essayez de redémarrer Ollama

## 📊 **Exigences système**

- **macOS** : 10.15 (Catalina) ou plus récent
- **RAM** : 8GB minimum (16GB recommandé)
- **Espace disque** : 5GB pour le modèle
- **Connexion internet** : Pour télécharger le modèle

## 🎯 **Test de l'installation**

```bash
# Test complet
python test_qa_generation.py

# Si tout fonctionne, lancez le script principal
python main.py
```
