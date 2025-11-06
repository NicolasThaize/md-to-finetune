# Traitement et Recherche Sémantique de Documents Markdown

Ce projet utilise **LlamaIndex** pour traiter et indexer un cours de droit administratif en Markdown, permettant une recherche sémantique intelligente.

## 🚀 Fonctionnalités

- **Chargement intelligent** : Parse les fichiers Markdown en préservant la hiérarchie (titres, sous-titres)
- **Découpage sémantique** : Crée des chunks logiques et cohérents
- **Indexation vectorielle** : Utilise des embeddings OpenAI pour la recherche sémantique
- **Recherche intelligente** : Trouve les passages les plus pertinents selon le sens
- **Interface interactive** : Permet de poser des questions en langage naturel

## 📋 Prérequis

1. **Python 3.8+**
2. **Clé API OpenAI** (pour les embeddings et le LLM)

## 🛠️ Installation

1. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

2. **Configurer la clé API OpenAI** :
```bash
export OPENAI_API_KEY="votre_cle_api_openai"
```

## 🎯 Utilisation

### Exécution du script principal

```bash
python main.py
```

### Fonctionnalités principales

1. **Chargement automatique** : Le script charge `sources/droitadminSmall.md`
2. **Indexation** : Crée un index vectoriel pour la recherche rapide
3. **Recherche sémantique** : Trouve les passages pertinents selon le sens
4. **Interface interactive** : Permet de poser des questions

### Exemples de requêtes

- "Qu'est-ce que l'administration?"
- "Définition des personnes publiques"
- "Différence entre établissement public et collectivité territoriale"
- "Service public et externalisation"

## 🏗️ Architecture

### Classe `MarkdownDocumentProcessor`

- **`load_and_parse_markdown()`** : Charge et parse le fichier Markdown
- **`_parse_markdown_hierarchy()`** : Préserve la structure hiérarchique
- **`create_chunks_with_hierarchy()`** : Découpe en chunks sémantiques
- **`build_index()`** : Construit l'index vectoriel
- **`search()`** : Effectue la recherche sémantique

### Découpage intelligent

- **Préservation de la hiérarchie** : Maintient les relations titre/contenu
- **Chunks sémantiques** : Découpage logique par sections
- **Métadonnées enrichies** : Titre parent, niveau, type de section

### Indexation vectorielle

- **Embeddings OpenAI** : `text-embedding-3-small`
- **Stockage persistant** : Sauvegarde dans `./storage/`
- **Recherche rapide** : Similarité cosinus optimisée

## 📊 Résultats

Le script affiche :
- **Score de pertinence** pour chaque résultat
- **Titre de la section** et hiérarchie
- **Extrait du contenu** pertinent
- **Métadonnées** complètes

## 🔧 Configuration

### Paramètres modifiables

```python
processor = MarkdownDocumentProcessor(
    markdown_file_path="chemin/vers/fichier.md",
    storage_dir="./storage",
    chunk_size=1024,        # Taille des chunks
    chunk_overlap=200       # Chevauchement entre chunks
)
```

### Modèles utilisés

- **Embeddings** : `text-embedding-3-small` (OpenAI)
- **LLM** : `gpt-3.5-turbo` (OpenAI)
- **Parser** : `MarkdownNodeParser` (LlamaIndex)

## 📁 Structure des fichiers

```
txt-to-chunks/
├── main.py                 # Script principal
├── requirements.txt        # Dépendances Python
├── README.md              # Documentation
├── sources/
│   └── droitadminSmall.md # Fichier source
└── storage/               # Index vectoriel (généré)
```

## 🚨 Notes importantes

1. **Clé API requise** : Nécessite une clé API OpenAI valide
2. **Première exécution** : Plus lente (construction de l'index)
3. **Exécutions suivantes** : Plus rapides (index chargé)
4. **Stockage** : L'index est sauvegardé dans `./storage/`

## 🔍 Exemples de recherche

Le script inclut des exemples de recherche sémantique :

- Recherche par concepts
- Recherche par définitions
- Recherche par comparaisons
- Recherche par processus

## 📈 Performance

- **Indexation** : ~30-60 secondes (première fois)
- **Recherche** : ~1-3 secondes par requête
- **Stockage** : ~10-50 MB selon la taille du document
