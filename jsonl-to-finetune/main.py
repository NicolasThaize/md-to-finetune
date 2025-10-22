"""CLI interface for JSONL to fine-tune pipeline."""

import argparse
import sys
from pathlib import Path

from core.config import TrainingConfig, LoRAConfig
from core.pipeline import FineTuningPipeline


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-OSS-20B with LoRA using JSONL data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning
  python main.py data.jsonl

  # Custom configuration
  python main.py data.jsonl --epochs 5 --batch-size 8 --learning-rate 1e-4

  # Validate JSONL only
  python main.py data.jsonl --validate-only
        """
    )
    
    # Required arguments
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to JSONL training file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetuned_model"),
        help="Output directory for fine-tuned model (default: finetuned_model)"
    )
    
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Base model to fine-tune (default: openai/gpt-oss-20b)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )
    
    # Other options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate JSONL format, don't train"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 training (default: True)"
    )
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file
    if not args.jsonl_file.exists():
        print(f"Error: JSONL file not found: {args.jsonl_file}")
        sys.exit(1)
    
    # Create configuration
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_config=lora_config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        fp16=args.fp16
    )
    
    # Create pipeline
    pipeline = FineTuningPipeline(config)
    
    try:
        if args.validate_only:
            print("Validating JSONL format...")
            if pipeline.validate_jsonl(args.jsonl_file):
                print("✅ JSONL format is valid!")
            else:
                print("❌ JSONL format validation failed!")
                sys.exit(1)
        else:
            # Run full pipeline
            output_dir = pipeline.run(args.jsonl_file)
            print(f"\n🎉 Fine-tuning completed successfully!")
            print(f"📁 Model saved to: {output_dir}")
            print(f"💡 You can now use the fine-tuned model for inference!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
