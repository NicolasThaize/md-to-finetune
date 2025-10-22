"""Tokenizer management for fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer


class TokenizerManager:
    """Manages tokenizer for fine-tuning."""
    
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Optional[AutoTokenizer] = None
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        
        return self._tokenizer
    
    def tokenize_examples(self, examples: List[Dict[str, str]]) -> Dict[str, List]:
        """Tokenize training examples."""
        questions = [ex["question"] for ex in examples]
        answers = [ex["answer"] for ex in examples]
        
        # Tokenize questions and answers
        question_encodings = self.tokenizer(
            questions,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        answer_encodings = self.tokenizer(
            answers,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": question_encodings["input_ids"],
            "attention_mask": question_encodings["attention_mask"],
            "labels": answer_encodings["input_ids"]
        }
    
    def create_prompt(self, question: str, answer: str = "") -> str:
        """Create a formatted prompt for training."""
        if answer:
            return f"Question: {question}\nAnswer: {answer}"
        else:
            return f"Question: {question}\nAnswer:"
    
    def save_tokenizer(self, output_dir: Path) -> None:
        """Save tokenizer to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
