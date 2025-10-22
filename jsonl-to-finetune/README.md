# JSONL to Fine-tune Pipeline

Pipeline de fine-tuning pour GPT-OSS-20B utilisant LoRA (Low-Rank Adaptation) avec des données au format JSONL.

## Architecture

Le projet suit les patterns de développement suivants :

- **MVC (Model-View-Controller)** : Séparation claire des responsabilités
- **Observer Pattern** : Notifications de progression d'entraînement
- **Strategy Pattern** : Configuration flexible de LoRA
- **Factory Pattern** : Création de composants modulaires

## Structure

```
jsonl-to-finetune/
├── main.py                 # Point d'entrée CLI
├── requirements.txt        # Dépendances
├── README.md              # Documentation
├── data/                  # Couche de données
│   ├── __init__.py
│   └── jsonl_loader.py    # Chargement et validation JSONL
└── core/                  # Logique métier
    ├── __init__.py
    ├── config.py         # Configuration LoRA et training
    ├── tokenizer.py      # Gestion du tokenizer
    ├── lora_trainer.py   # Entraînement LoRA
    └── pipeline.py       # Orchestrateur principal
```

## Installation

```bash
cd jsonl-to-finetune
pip install -r requirements.txt
```

## Utilisation

### Exemple basique

```bash
python main.py training_data.jsonl
```

### Configuration avancée

```bash
python main.py training_data.jsonl \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --output-dir my_finetuned_model
```

### Validation uniquement

```bash
python main.py training_data.jsonl --validate-only
```

## Format JSONL

Le fichier JSONL doit contenir des exemples au format :

```jsonl
{"question": "Qu'est-ce que l'IA ?", "answer": "L'intelligence artificielle est..."}
{"user": "Comment fonctionne un réseau de neurones ?", "assistant": "Un réseau de neurones..."}
```

## Configuration LoRA

- **r** : Rang de décomposition (défaut: 16)
- **lora_alpha** : Facteur d'échelle (défaut: 32)
- **lora_dropout** : Dropout (défaut: 0.1)
- **target_modules** : Modules ciblés (défaut: ["q_proj", "v_proj"])

## Exemple de données

```jsonl
{"question": "Qu'est-ce que Python ?", "answer": "Python est un langage de programmation..."}
{"question": "Comment installer pip ?", "answer": "Pour installer pip, utilisez..."}
{"question": "Qu'est-ce que Git ?", "answer": "Git est un système de contrôle de version..."}
```

## Sortie

Le modèle fine-tuné est sauvegardé dans le répertoire spécifié avec :
- Modèle LoRA adapté
- Tokenizer
- Configuration d'entraînement
- Métadonnées

## Bonnes pratiques

1. **Validation** : Toujours valider le JSONL avant l'entraînement
2. **Configuration** : Ajustez les hyperparamètres selon vos données
3. **Monitoring** : Surveillez la perte pendant l'entraînement
4. **Sauvegarde** : Le modèle est sauvegardé automatiquement
