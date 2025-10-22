# 🎯 Générateur de Dataset Q/R

Générateur optimisé pour créer des datasets d'entraînement Q/R à partir de documents Markdown, sans fonctionnalités de recherche sémantique.

## 🏗️ Architecture

### Design Patterns Implémentés

#### **1. Strategy Pattern**
- **`QuestionGenerator`** et **`AnswerGenerator`** : Interfaces pour différents algorithmes de génération
- **`LLMQuestionGenerator`** et **`LLMAnswerGenerator`** : Implémentations concrètes utilisant Ollama

#### **2. Factory Pattern**
- **`DatasetGeneratorFactory`** : Création centralisée des générateurs avec configuration par défaut

#### **3. Facade Pattern**
- **`DatasetGenerator`** : Interface simplifiée pour l'ensemble du processus

#### **4. Abstract Factory Pattern**
- **`MarkdownParser`**, **`ChunkProcessor`**, **`DatasetExporter`** : Interfaces pour différents types de traitement

## 📁 Structure du Code

```
dataset_generator.py
├── QAPair (dataclass)           # Représentation d'une paire Q/R
├── MarkdownParser (interface)   # Parsing de documents
├── HierarchicalMarkdownParser   # Parser préservant la hiérarchie
├── ChunkProcessor (interface)   # Traitement des chunks
├── MarkdownChunkProcessor       # Processeur spécialisé Markdown
├── QuestionGenerator (interface) # Génération de questions
├── LLMQuestionGenerator         # Générateur utilisant LLM
├── AnswerGenerator (interface)   # Génération de réponses
├── LLMAnswerGenerator           # Générateur utilisant LLM
├── QAPairGenerator              # Orchestrateur Q/R
├── DatasetExporter (interface)    # Export de datasets
├── JSONLExporter                # Export au format JSONL
├── DatasetGenerator             # Générateur principal
└── DatasetGeneratorFactory      # Factory de création
```

## 🚀 Utilisation

### Installation

```bash
# Dépendances
pip install -r requirements.txt

# Ollama (pour le LLM local)
./install_macos.sh  # Sur macOS
# ou
./install_ollama.sh  # Sur Linux
```

### Utilisation Simple

```bash
# Génération automatique
python dataset_generator.py
```

### Utilisation Avancée

```python
from dataset_generator import DatasetGeneratorFactory

# Créer un générateur personnalisé
generator = DatasetGeneratorFactory.create_default_generator("mistral:7b-instruct")

# Générer le dataset
qa_pairs = generator.generate_dataset(
    markdown_file=Path("sources/droitadminSmall.md"),
    output_file=Path("training_data.jsonl")
)
```

## 🧪 Tests

```bash
# Tests complets
python test_dataset_generator.py

# Tests spécifiques
python -c "from test_dataset_generator import test_imports; test_imports()"
```

## 📊 Format de Sortie

### JSONL Standard

Chaque ligne contient une paire Q/R au format :

```json
{"messages": [{"role": "user", "content": "Qu'est-ce que l'administration?"}, {"role": "assistant", "content": "L'administration est l'ensemble des personnes publiques françaises."}]}
```

### Métadonnées

- **`source_title`** : Titre de la section source
- **`source_level`** : Niveau hiérarchique (1-6)
- **`question`** : Question générée
- **`answer`** : Réponse basée sur le contenu

## 🔧 Configuration

### Variables d'Environnement

```bash
# .env
OLLAMA_HOST=http://localhost:11434
```

### Paramètres LLM

```python
# Personnalisation du modèle
generator = DatasetGeneratorFactory.create_default_generator("llama2:7b")
```

## 📈 Performance

### Optimisations

- **Pas d'index vectoriel** : Suppression des fichiers inutiles
- **Parsing hiérarchique** : Préservation de la structure Markdown
- **Génération par chunks** : Traitement optimisé des sections
- **Export direct** : Pas de stockage intermédiaire

### Métriques

- **Temps de génération** : ~2-5 minutes pour 13 sections
- **Taille du dataset** : ~50-100 paires Q/R
- **Espace disque** : Seulement le fichier JSONL final

## 🎯 Avantages

### vs Script Original

| Aspect | Script Original | Script Optimisé |
|--------|----------------|-----------------|
| **Fichiers générés** | 5+ fichiers | 1 fichier JSONL |
| **Espace disque** | ~472MB | ~1MB |
| **Complexité** | Recherche + Q/R | Q/R uniquement |
| **Performance** | Lente (index) | Rapide (direct) |
| **Maintenance** | Complexe | Simple |

### Design Patterns

- **Séparation des responsabilités** : Chaque classe a un rôle précis
- **Extensibilité** : Facile d'ajouter de nouveaux générateurs
- **Testabilité** : Chaque composant peut être testé indépendamment
- **Réutilisabilité** : Composants modulaires

## 🔍 Exemples d'Usage

### Génération Simple

```python
from dataset_generator import DatasetGeneratorFactory

generator = DatasetGeneratorFactory.create_default_generator()
qa_pairs = generator.generate_dataset(
    Path("mon_document.md"),
    Path("mon_dataset.jsonl")
)
```

### Génération Personnalisée

```python
from dataset_generator import (
    HierarchicalMarkdownParser,
    MarkdownChunkProcessor,
    LLMQuestionGenerator,
    LLMAnswerGenerator,
    QAPairGenerator,
    JSONLExporter,
    DatasetGenerator
)

# Composants personnalisés
parser = HierarchicalMarkdownParser()
processor = MarkdownChunkProcessor()
question_gen = LLMQuestionGenerator("custom-model")
answer_gen = LLMAnswerGenerator("custom-model")
qa_gen = QAPairGenerator(question_gen, answer_gen)
exporter = JSONLExporter()

# Générateur personnalisé
generator = DatasetGenerator(parser, processor, qa_gen, exporter)
```

## 🚨 Dépannage

### Erreurs Communes

1. **"Ollama not found"**
   ```bash
   # Vérifier qu'Ollama est installé et lancé
   ollama list
   ```

2. **"Model not found"**
   ```bash
   # Installer le modèle
   ollama pull mistral:7b-instruct
   ```

3. **"Import errors"**
   ```bash
   # Réinstaller les dépendances
   pip install -r requirements.txt
   ```

### Logs

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📚 Documentation Technique

### Interfaces Principales

#### `MarkdownParser`
```python
class MarkdownParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> List[Document]:
        pass
```

#### `QuestionGenerator`
```python
class QuestionGenerator(ABC):
    @abstractmethod
    def generate_questions(self, chunk: Document) -> List[str]:
        pass
```

#### `AnswerGenerator`
```python
class AnswerGenerator(ABC):
    @abstractmethod
    def generate_answer(self, question: str, chunk: Document) -> str:
        pass
```

### Extensibilité

Pour ajouter de nouveaux types de générateurs :

1. **Implémenter l'interface** correspondante
2. **Créer une factory** personnalisée
3. **Tester** avec les tests unitaires

## 🎉 Résultat

Un script optimisé, maintenable et extensible pour la génération de datasets Q/R, sans les complexités inutiles de la recherche sémantique.
