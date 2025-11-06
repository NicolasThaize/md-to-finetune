import argparse
import logging
from pathlib import Path

from pipeline.factory import DatasetGeneratorFactory


def run() -> None:
    parser = argparse.ArgumentParser(description="Générateur de dataset Q/R")
    parser.add_argument(
        "--format",
        choices=["messages", "question_answer", "user_assistant", "mistral_template"],
        default="messages",
        help="Format de sortie (défaut: messages)",
    )
    parser.add_argument(
        "--mistral-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Modèle Mistral pour le tokenizer (si format=mistral_template)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sources/droitadminSmall.md"),
        help="Fichier Markdown d'entrée",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier de sortie (extension ajustée automatiquement selon le format)",
    )
    parser.add_argument(
        "--llm-model",
        default="mistral:7b-instruct",
        help="Modèle Ollama pour la génération Q/R",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    output_file = args.output
    if not output_file:
        output_file = Path("training_data.csv" if args.format == "mistral_template" else "training_data.jsonl")

    generator = DatasetGeneratorFactory.create(
        llm_model=args.llm_model,
        output_format=args.format,
        mistral_model_name=args.mistral_model,
    )
    generator.generate_dataset(args.input, output_file)


if __name__ == "__main__":
    run()

import argparse
from pathlib import Path

from pipeline.factory import DatasetGeneratorFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Générateur de dataset Q/R")
    parser.add_argument(
        "--format",
        choices=["messages", "question_answer", "user_assistant", "mistral_template"],
        default="messages",
        help="Format de sortie (défaut: messages)"
    )
    parser.add_argument(
        "--mistral-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Modèle Mistral pour le tokenizer (si format=mistral_template)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Fichier Markdown d'entrée"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Fichier de sortie (extension ajustée automatiquement selon le format)"
    )
    parser.add_argument(
        "--llm-model",
        default="mistral:7b-instruct",
        help="Modèle Ollama pour la génération Q/R"
    )
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = Path("training_data.csv" if args.format == "mistral_template" else "training_data.jsonl")

    generator = DatasetGeneratorFactory.create(
        llm_model=args.llm_model,
        output_format=args.format,
        mistral_model_name=args.mistral_model
    )
    qa_pairs = generator.generate_dataset(args.input, output_file)
    print(f"\n✅ Dataset généré avec succès!")
    print(f"📊 {len(qa_pairs)} paires Q/R créées")
    print(f"💾 Fichier: {output_file}")
    print(f"📝 Format: {args.format}")


if __name__ == "__main__":
    main()


