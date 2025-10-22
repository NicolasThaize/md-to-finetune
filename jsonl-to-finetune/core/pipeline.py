"""Main fine-tuning pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from data.jsonl_loader import JSONLLoader, JSONLTrainingExample
from .config import TrainingConfig
from .tokenizer import TokenizerManager
from .lora_trainer import LoRATrainer, TrainingObserver


class PipelineObserver(TrainingObserver):
    """Simple observer for pipeline progress."""
    
    def __init__(self) -> None:
        self.current_step = 0
        self.current_loss = 0.0
    
    def on_training_start(self, config: TrainingConfig) -> None:
        print(f"Starting fine-tuning with {config.num_epochs} epochs...")
        print(f"Base model: {config.base_model}")
        print(f"Output directory: {config.output_dir}")
    
    def on_training_progress(self, step: int, loss: float) -> None:
        self.current_step = step
        self.current_loss = loss
        if step % 10 == 0:  # Print every 10 steps
            print(f"Step {step}, Loss: {loss:.4f}")
    
    def on_training_complete(self, output_dir: Path) -> None:
        print(f"Training completed! Final loss: {self.current_loss:.4f}")
        print(f"Model saved to: {output_dir}")


class FineTuningPipeline:
    """Main pipeline for fine-tuning GPT-OSS-20B with LoRA."""
    
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.tokenizer_manager = TokenizerManager(
            model_name=config.base_model,
            max_length=config.max_length
        )
        self.trainer = LoRATrainer(config, self.tokenizer_manager)
        self.observer = PipelineObserver()
        self.trainer.add_observer(self.observer)
    
    def run(self, jsonl_path: Path) -> Path:
        """Run the complete fine-tuning pipeline."""
        print("=" * 50)
        print("JSONL to Fine-tune Pipeline")
        print("=" * 50)
        
        # Step 1: Load and validate JSONL
        print("\n1. Loading JSONL data...")
        loader = JSONLLoader(jsonl_path)
        
        if not loader.validate_format():
            raise ValueError("JSONL format validation failed")
        
        examples = loader.load_examples()
        print(f"Loaded {len(examples)} training examples")
        
        # Step 2: Convert to training format
        print("\n2. Preparing training data...")
        training_data = [ex.to_dict() for ex in examples]
        
        # Step 3: Train with LoRA
        print("\n3. Starting LoRA fine-tuning...")
        self.trainer.train(training_data)
        
        return self.config.output_dir
    
    def validate_jsonl(self, jsonl_path: Path) -> bool:
        """Validate JSONL file without training."""
        try:
            loader = JSONLLoader(jsonl_path)
            return loader.validate_format()
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
