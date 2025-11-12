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
main.py                             # Pointe vers la CLI (shim)

cli/
└── main.py                         # argparse → factory + pipeline

core/
├── domain.py                       # QAPair (dataclass)
├── parsing.py                      # MarkdownParser, HierarchicalMarkdownParser
├── chunking.py                     # ChunkProcessor, MarkdownChunkProcessor
└── qa.py                           # QuestionGenerator, AnswerGenerator

llm/
└── ollama_generators.py            # BaseFRQuestionGenerator, BaseFRAnswerGenerator

pipeline/
├── generator.py                    # QAPairGenerator, DatasetGenerator (facade)
└── factory.py                      # DatasetGeneratorFactory (assemblage)

exporters/
├── base.py                         # Interface DatasetExporter
├── messages.py                     # MessagesFormatExporter (JSONL)
├── question_answer.py              # QuestionAnswerFormatExporter (JSONL)
├── user_assistant.py               # UserAssistantFormatExporter (JSONL)
└── mistral_template.py             # MistralChatTemplateExporter (JSONL)
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
# JSONL (messages)
python main.py --format messages --input sources/droitadminSmall.md --output training_data.jsonl

# JSONL (Mistral chat template)
python main.py --format mistral_template --input sources/droitadminSmall.md --output training_data.jsonl --mistral-model mistralai/Mistral-7B-v0.1
```

### Utilisation Avancée

```python
from pathlib import Path
from pipeline.factory import DatasetGeneratorFactory

# Générateur JSONL (messages)
generator = DatasetGeneratorFactory.create(
    llm_model="mistral:7b-instruct",
    output_format="messages",
)
qa_pairs = generator.generate_dataset(
    markdown_file=Path("sources/droitadminSmall.md"),
    output_file=Path("training_data.jsonl")
)

# Générateur Mistral chat template
generator_mistral = DatasetGeneratorFactory.create(
    llm_model="mistral:7b-instruct",
    output_format="mistral_template",
    mistral_model_name="mistralai/Mistral-7B-v0.1",
)
qa_pairs = generator_mistral.generate_dataset(
    markdown_file=Path("sources/droitadminSmall.md"),
    output_file=Path("training_data.jsonl")
)
```

## 🧪 Tests

```bash
# Tests complets
python test_main.py

# Tests spécifiques
python -c "from test_dataset_generator import test_imports; test_imports()"
```

## 📊 Format de Sortie

### Formats pris en charge

- Messages (JSONL):
  ```json
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```

- Question/Answer (JSONL):
  ```json
  {"question": "...", "answer": "..."}
  ```

- User/Assistant (JSONL):
  ```json
  {"user": "...", "assistant": "..."}
  ```

- Mistral chat template (JSONL):
  ```json
  {"text": "<s>[INST] Ma question [/INST] réponse </s>"}
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
from pipeline.factory import DatasetGeneratorFactory

generator = DatasetGeneratorFactory.create()
qa_pairs = generator.generate_dataset(
    Path("mon_document.md"),
    Path("mon_dataset.jsonl")
)
```

### Génération Personnalisée

```python
from core.parsing import HierarchicalMarkdownParser
from core.chunking import MarkdownChunkProcessor
from llm.ollama_generators import BaseFRQuestionGenerator, BaseFRAnswerGenerator
from pipeline.generator import QAPairGenerator, DatasetGenerator
from exporters.messages import MessagesFormatExporter

# Composants personnalisés
parser = HierarchicalMarkdownParser()
processor = MarkdownChunkProcessor()
question_gen = BaseFRQuestionGenerator("custom-model")
answer_gen = BaseFRAnswerGenerator("custom-model")
qa_gen = QAPairGenerator(question_gen, answer_gen)
exporter = MessagesFormatExporter()

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
