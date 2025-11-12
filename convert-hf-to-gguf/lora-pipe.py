import argparse
from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAMerger:
    """Utility to merge a LoRA adapter into a base Hugging Face model."""

    def __init__(self, base_model: str, adapter_path: str, output_path: str):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.output_path = Path(output_path)

    def merge_and_save(self) -> Path:
        """Merge the LoRA adapter into the base model and save the result."""
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        peft_model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        merged_model = peft_model.merge_and_unload()

        self.output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(self.output_path))
        tokenizer.save_pretrained(str(self.output_path))

        return self.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a base Hugging Face model and save the merged weights."
    )
    parser.add_argument(
        "--basemodel",
        required=True,
        help="Base Hugging Face model identifier or local path.",
    )
    parser.add_argument(
        "--tunedmodel",
        required=True,
        help="Local path to the fine-tuned LoRA adapter.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for the merged model and tokenizer.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> Path:
    if args is None:
        args = parse_args()

    merger = LoRAMerger(
        base_model=args.basemodel,
        adapter_path=args.tunedmodel,
        output_path=args.output,
    )
    return merger.merge_and_save()


if __name__ == "__main__":
    merge_path = main()
    print(f"Merged model saved to: {merge_path}")