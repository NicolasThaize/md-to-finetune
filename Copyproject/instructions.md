## Contexte

Outil de fine-tuning de modèles LLM basé sur HuggingFace pour la logique et Tkinter pour l'interface desktop. L'utilisateur choisit un modèle pré-entraîné, dépose un fichier TXT et lance une pipeline de fine-tuning. Une barre d'état affiche les logs/états.

## Philosophie

- Simplicité d'abord: une UI claire, des actions explicites, peu d'options au départ.
- Séparation stricte des responsabilités (MVC) pour faciliter l'évolution.
- Extensibilité: patterns (Factory, Observer, Strategy) pour ajouter modèles/stratégies sans casser l'existant.
- Transparence: logs lisibles et statut en temps réel dans l'UI.
- Sécurité minimale: valider les chemins, formats et erreurs courantes.

## Architecture et patterns

- MVC
  - View: `ui/` (Tkinter `MainWindow`, `StatusBar`)
  - Controller: callbacks injectés du `main.py` vers la View et le Core
  - Model (business): `core/` (managers, pipeline, config)
- Factory Pattern
  - `ModelFactory` pour instancier des modèles HuggingFace (ou placeholders). Ajoutez/inscrivez de nouveaux builders sans modifier l'appelant.
- Observer Pattern
  - `ModelManager`/`TrainingPipeline` notifient l'UI des changements/états (méthodes `on_status...`).
- Strategy Pattern
  - `TrainingStrategy` définit l'interface; implémentations concrètes (ex: `DummyTrainingStrategy`, puis stratégie HF Trainer/PEFT).

## Structure du projet

```
Copyproject/
├── main.py                  # Point d'entrée
├── ui/
│   ├── main_window.py       # Fenêtre principale (View)
│   └── status_bar.py        # Barre d'état/logs
├── core/
│   ├── model_manager.py     # Gestion des modèles (Factory + Observer)
│   ├── data_manager.py      # Fichiers TXT (Data layer)
│   ├── training_pipeline.py # Pipeline de fine-tuning (Strategy + Observer)
│   └── config.py            # Configuration et chemins
├── data/
│   ├── input/               # Fichiers TXT
│   └── output/              # Modèles fine-tunés
├── logs/                    # Fichiers de logs
├── requirements.txt         # Dépendances
└── instructions.md          # Ce document
```

## Mise en œuvre (étapes clés)

1) UI (Tkinter)
- Boutons: sélectionner modèle, uploader TXT, lancer le fine-tuning.
- `StatusBar` met à jour les messages via Observer.

2) Logique (Core)
- `ModelManager`: sélection/chargement de modèles via `ModelFactory` + notifications.
- `DataManager`: validation/copier des `.txt`, lecture des lignes.
- `TrainingPipeline`: orchestre la stratégie de fine-tuning et notifie l'UI.

3) Données
- `data/input`: sources TXT
- `data/output`: sauvegardes modèles/poids
- `logs`: fichiers de logs (rotations à envisager plus tard)

## Bonnes pratiques (à suivre par toutes les IA)

- Respecter strictement MVC et patterns existants; ne pas mélanger UI et Core.
- Ajouter de nouvelles stratégies de training en implémentant `TrainingStrategy` (pas de `if/else` géants).
- Étendre les modèles via `ModelFactory.register(name, builder)`.
- Reporter tout état vers l'UI via les méthodes Observer (`on_status...`) plutôt que des `print`.
- Valider les chemins/fichiers et lever des erreurs claires (messages destinés à l'UI).
- Garder les dépendances minimales et versionnées dans `requirements.txt`.
- Préférer des noms explicites et code lisible (pas d’abréviations obscures).
- Éviter les effets de bord; fonctions pures quand c'est possible.
- Journaliser les erreurs/états importants; ne pas ignorer les exceptions.

## Dépendances et exécution

- Installer depuis `requirements.txt` (Py3.10+ conseillé).
- Lancer l'application:
```bash
python Copyproject/main.py
```

## Évolutions prévues (prochaines étapes simples)

- Faire en sorte que les modèles dans le dossier data/output soient sélectionnables dans le selectionneur de modèles en indiquant qu'ils viennent du local.
- Implémenter une stratégie HuggingFace Trainer (datasets/tokenizer/Trainer).
- Ajout d’une sélection de modèle (liste prédéfinie + champ libre HF Hub).
- Affichage des logs en temps réel dans la fenêtre centrale.
- Gestion des erreurs utilisateur (fichiers vides, encodage, etc.).


