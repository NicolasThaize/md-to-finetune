"""Configuration for fine-tuning pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: list[str] = None  # Will be set to ["q_proj", "v_proj"] for GPT
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    base_model: str = "openai/gpt-oss-20b"
    output_dir: Path = Path("finetuned_model")
    
    # LoRA
    lora_config: LoRAConfig = None
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Other
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    
    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = LoRAConfig()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
