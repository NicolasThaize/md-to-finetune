# Local Embedder - Recherche Sémantique et Génération Q/R Locale

Ce projet utilise des **embeddings locaux** pour la recherche sémantique dans des documents Markdown, et génère automatiquement des **paires question/réponse** pour l'entraînement de modèles, sans dépendre d'APIs externes coûteuses.

## 🚀 **Avantages de la solution locale**

- ✅ **Gratuit** : Pas de coûts d'API
- ✅ **Sans limites** : Pas de rate limits
- ✅ **Rapide** : Pas de latence réseau
- ✅ **Privé** : Données restent locales
- ✅ **Contrôle total** : Personnalisation possible
- ✅ **Génération Q/R** : Création automatique de datasets d'entraînement

## 🤖 **Modèles utilisés**

### **Embeddings**
**`sentence-transformers/distiluse-base-multilingual-cased`**
- 🌍 **Multilingue** : Optimisé pour le français
- 💾 **Taille** : ~471MB
- 🧠 **Dimension** : 512
- ⚡ **Performance** : Excellente qualité

### **LLM pour génération Q/R**
**`mistral:7b-instruct`**
- 🧠 **Modèle** : Mistral 7B Instruct
- 💾 **Taille** : ~4.1GB
- 🌍 **Multilingue** : Excellent en français
- ⚡ **Performance** : Très bonne qualité

## 📋 **Prérequis**

1. **Python 3.8+**
2. **RAM** : 8GB minimum (16GB recommandé pour le LLM)
3. **GPU** : Optionnel mais recommandé pour la vitesse
4. **Espace disque** : 6GB pour les modèles et l'index
5. **Ollama** : Pour le LLM local (Mistral 7B)

## 🛠️ **Installation**

### 1. **Installer les dépendances**

```bash
cd local-embedder
pip install -r requirements.txt
```

### 2. **Installation d'Ollama et du modèle Mistral**

#### **Sur macOS (recommandé)**
```bash
# Installation automatique pour macOS
./install_macos.sh

# Ou installation manuelle
# Voir INSTALL_MACOS.md pour le guide détaillé
```

#### **Sur Linux**
```bash
# Installation automatique
./install_ollama.sh

# Ou installation manuelle
# 1. Téléchargez Ollama: https://ollama.ai
# 2. Installez le modèle: ollama pull mistral:7b-instruct
```

### 3. **Installation GPU (optionnel mais recommandé)**

Pour utiliser le GPU et accélérer les calculs :

```bash
# Pour CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🎯 **Utilisation**

### **Test de la configuration**

```bash
# Test des embeddings
python test_local.py

# Test de la génération Q/R
python test_qa_generation.py
```

### **Exécution du script principal**

```bash
python main.py
```

Le script propose un menu interactif avec :
1. **🔍 Recherche sémantique** : Questions sur le document
2. **🤖 Génération Q/R** : Création automatique de paires question/réponse
3. **📊 Statistiques** : Informations sur le dataset
4. **❌ Quitter**

## 🤖 **Génération de données d'entraînement**

### **Format JSONL généré**

Chaque ligne du fichier `training_data.jsonl` contient une paire question/réponse :

```json
{"messages": [{"role": "user", "content": "Qu'est-ce que l'administration?"}, {"role": "assistant", "content": "L'administration est l'ensemble des personnes publiques françaises."}]}
```

### **Processus de génération**

1. **Parsing** : Découpage du document en chunks sémantiques
2. **Génération de questions** : 2-3 questions par chunk via Mistral 7B
3. **Génération de réponses** : Réponses basées sur le contenu du chunk
4. **Formatage** : Conversion au format JSONL standard

### **Exemple d'utilisation**

```bash
# Générer le dataset complet
python main.py
# Choisir l'option 2

# Le fichier sera créé dans: storage/training_data.jsonl
```

## 🔧 **Configuration**

### **Modèles d'embedding disponibles**

| Modèle | Taille | RAM | Performance | Français |
|--------|--------|-----|-------------|----------|
| `all-MiniLM-L6-v2` | 22MB | 1GB | ⭐⭐⭐ | ⭐⭐⭐ |
| `paraphrase-multilingual` | 118MB | 2GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **`distiluse-base-multilingual`** | **471MB** | **4GB** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** |

### **Configuration personnalisée**

```python
processor = MarkdownDocumentProcessorLocal(
    markdown_file_path="chemin/vers/fichier.md",
    storage_dir="./storage",
    embedding_model="sentence-transformers/distiluse-base-multilingual-cased",
    chunk_size=1024,
    chunk_overlap=200
)
```

## 📊 **Performance**

### **Première exécution**
- **Téléchargement du modèle** : 2-5 minutes
- **Génération des embeddings** : 1-3 minutes
- **Total** : 3-8 minutes

### **Exécutions suivantes**
- **Chargement de l'index** : 5-10 secondes
- **Recherche** : 0.1-0.5 secondes

## 🖥️ **Support GPU**

### **Vérification GPU**

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### **Configuration GPU automatique**

Le script détecte automatiquement la disponibilité du GPU et l'utilise si possible.

## 📁 **Structure des fichiers**

```
local-embedder/
├── main.py                 # Script principal
├── test_local.py          # Script de test
├── requirements.txt       # Dépendances
├── README.md             # Documentation
└── storage/              # Index et cache (généré)
    ├── index/            # Index vectoriel
    └── embeddings_cache/ # Cache des embeddings
```

## 🔍 **Exemples de recherche**

Le script inclut des exemples de recherche sémantique :

- "Qu'est-ce que l'administration?"
- "Définition des personnes publiques"
- "Différence entre établissement public et collectivité territoriale"
- "Service public et externalisation"

## ⚡ **Optimisations**

### **Cache des embeddings**
Les embeddings sont mis en cache pour éviter les recalculs.

### **Quantification (optionnel)**
```python
# Pour réduire l'utilisation mémoire
model_kwargs={"torch_dtype": "float16"}
```

### **Batch processing**
Le script traite les documents par batch pour optimiser la mémoire.

## 🚨 **Dépannage**

### **Erreur de mémoire**
- Réduisez `chunk_size` (512 au lieu de 1024)
- Utilisez un modèle plus petit
- Fermez d'autres applications

### **Erreur de téléchargement**
- Vérifiez votre connexion internet
- Le modèle sera téléchargé automatiquement

### **Performance lente**
- Installez PyTorch avec support GPU
- Utilisez un modèle plus petit si nécessaire

## 📈 **Comparaison avec les APIs**

| Solution | Coût | Vitesse | Limites | Privé |
|----------|------|---------|---------|-------|
| **Local** | Gratuit | Rapide | Aucune | ✅ |
| OpenAI | Payant | Rapide | Rate limits | ❌ |
| Mistral | Payant | Rapide | Rate limits | ❌ |

## 🎯 **Recommandations**

1. **Première utilisation** : Lancez `test_local.py` pour vérifier la configuration
2. **GPU recommandé** : Pour des performances optimales
3. **Modèle distiluse** : Meilleur équilibre qualité/performance pour le français
4. **Cache activé** : Les embeddings sont mis en cache automatiquement

## 🔧 **Personnalisation avancée**

### **Modèle personnalisé**
```python
# Utiliser un modèle différent
processor = MarkdownDocumentProcessorLocal(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### **Configuration GPU manuelle**
```python
# Forcer l'utilisation du CPU
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/distiluse-base-multilingual-cased",
    device="cpu"
)
```

Cette solution locale vous donne un contrôle total sur vos données et vos coûts !
