"""LoRA fine-tuning trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from .config import TrainingConfig
from .tokenizer import TokenizerManager


class TrainingObserver(Protocol):
    """Observer for training progress."""
    def on_training_start(self, config: TrainingConfig) -> None:
        ...
    
    def on_training_progress(self, step: int, loss: float) -> None:
        ...
    
    def on_training_complete(self, output_dir: Path) -> None:
        ...


class LoRATrainer:
    """LoRA fine-tuning trainer with observer pattern."""
    
    def __init__(self, config: TrainingConfig, tokenizer_manager: TokenizerManager) -> None:
        self.config = config
        self.tokenizer_manager = tokenizer_manager
        self.observers: List[TrainingObserver] = []
        self.model = None
        self.trainer = None
    
    def add_observer(self, observer: TrainingObserver) -> None:
        """Add training observer."""
        self.observers.append(observer)
    
    def _notify_training_start(self) -> None:
        """Notify observers of training start."""
        for observer in self.observers:
            observer.on_training_start(self.config)
    
    def _notify_progress(self, step: int, loss: float) -> None:
        """Notify observers of training progress."""
        for observer in self.observers:
            observer.on_training_progress(step, loss)
    
    def _notify_complete(self) -> None:
        """Notify observers of training completion."""
        for observer in self.observers:
            observer.on_training_complete(self.config.output_dir)
    
    def prepare_model(self) -> None:
        """Load and prepare model with LoRA."""
        print(f"Loading base model: {self.config.base_model}")
        
        # Determine dtype based on fp16 setting
        model_dtype = torch.float16 if self.config.fp16 else torch.float32
        
        # Load base model with consistent dtype and device mapping
        try:
            print(f"Loading model with dtype={model_dtype}, device_map='auto'")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("Model loaded successfully with device_map='auto'")
        except Exception as e:
            print(f"Failed to load with device_map='auto': {e}")
            print("Trying with device_map=None...")
            # Fallback to loading on CPU first, then move to GPU if available
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    dtype=model_dtype,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print("Model loaded successfully with device_map=None")
                # Move to GPU if available
                if torch.cuda.is_available():
                    print("Moving model to GPU...")
                    model = model.cuda()
                    print("Model moved to GPU successfully")
            except Exception as e2:
                print(f"Failed to load with device_map=None: {e2}")
                raise e2
        
        # Verify model was loaded successfully
        if model is None:
            raise RuntimeError("Failed to load model - model is None")
        
        print(f"Model loaded successfully. Model type: {type(model)}")
        
        # Force dtype conversion to ensure consistency
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            print("Model loaded with device mapping - ensuring dtype consistency")
            # For offloaded models, we need to be more careful
            for name, param in model.named_parameters():
                if param.dtype != model_dtype:
                    print(f"Converting {name} from {param.dtype} to {model_dtype}")
                    param.data = param.data.to(dtype=model_dtype)
        else:
            # Model is fully loaded - can safely convert dtype
            print("Model is fully loaded - converting dtype")
            model = model.to(dtype=model_dtype)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(model, lora_config)
        
        # Ensure LoRA model is also in correct dtype
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            print("LoRA model using device mapping - ensuring dtype consistency")
            for name, param in self.model.named_parameters():
                if param.dtype != model_dtype:
                    print(f"Converting LoRA {name} from {param.dtype} to {model_dtype}")
                    param.data = param.data.to(dtype=model_dtype)
        else:
            self.model = self.model.to(dtype=model_dtype)
        
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for training."""
        # Create prompts
        prompts = []
        for ex in examples:
            prompt = self.tokenizer_manager.create_prompt(ex["question"], ex["answer"])
            prompts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer_manager.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Ensure consistent dtype for tensors
        model_dtype = torch.float16 if self.config.fp16 else torch.float32
        
        # Create dataset with proper dtype handling
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"]  # For causal LM, labels = input_ids
        })
        
        return dataset
    
    def train(self, examples: List[Dict[str, str]]) -> None:
        """Train the model with LoRA."""
        self._notify_training_start()
        
        # Prepare model
        self.prepare_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset(examples)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            fp16=self.config.fp16,
            bf16=False,  # Explicitly disable bf16 to avoid dtype conflicts
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            ddp_find_unused_parameters=False,  # Avoid issues with offloaded models
            save_safetensors=True,  # Use safetensors for better compatibility
        )
        
        # Custom data collator to ensure dtype consistency
        class CustomDataCollator(DataCollatorForLanguageModeling):
            def __init__(self, tokenizer, mlm=False, fp16=True):
                super().__init__(tokenizer=tokenizer, mlm=mlm)
                self.fp16 = fp16
            
            def __call__(self, features):
                batch = super().__call__(features)
                # Ensure all tensors are in the correct dtype
                model_dtype = torch.float16 if self.fp16 else torch.float32
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(dtype=torch.long if key in ['input_ids', 'labels', 'attention_mask'] else model_dtype)
                return batch
        
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer_manager.tokenizer,
            mlm=False,
            fp16=self.config.fp16
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer_manager.tokenizer,
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        
        # Save model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer_manager.save_tokenizer(self.config.output_dir)
        
        # Save config
        config_dict = {
            "base_model": self.config.base_model,
            "lora_config": {
                "r": self.config.lora_config.r,
                "lora_alpha": self.config.lora_config.lora_alpha,
                "target_modules": self.config.lora_config.target_modules,
                "lora_dropout": self.config.lora_config.lora_dropout,
            },
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
            }
        }
        
        with open(self.config.output_dir / "training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        self._notify_complete()
        print(f"Training complete! Model saved to: {self.config.output_dir}")
