"""JSONL data loader and validator for fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterator, Protocol


class TrainingExample(Protocol):
    """Protocol for training examples."""
    question: str
    answer: str


class JSONLTrainingExample:
    """Concrete training example from JSONL."""
    
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer
    
    def to_dict(self) -> Dict[str, str]:
        return {"question": self.question, "answer": self.answer}


class JSONLLoader:
    """Loads and validates JSONL training data."""
    
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    def load_examples(self) -> List[JSONLTrainingExample]:
        """Load all examples from JSONL file."""
        examples = []
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    example = self._parse_example(data, line_num)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Error parsing line {line_num}: {e}")
        
        if not examples:
            raise ValueError("No valid examples found in JSONL file")
        
        return examples
    
    def _parse_example(self, data: Dict, line_num: int) -> JSONLTrainingExample:
        """Parse a single example from JSON data."""
        if not isinstance(data, dict):
            raise ValueError(f"Line {line_num}: Expected dict, got {type(data)}")
        
        # Support both "question/answer" and "user/assistant" formats
        if "question" in data and "answer" in data:
            question = data["question"]
            answer = data["answer"]
        elif "user" in data and "assistant" in data:
            question = data["user"]
            answer = data["assistant"]
        else:
            raise ValueError(f"Line {line_num}: Missing required fields. Expected 'question/answer' or 'user/assistant'")
        
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"Line {line_num}: question and answer must be strings")
        
        if not question.strip() or not answer.strip():
            raise ValueError(f"Line {line_num}: question and answer cannot be empty")
        
        return JSONLTrainingExample(question.strip(), answer.strip())
    
    def validate_format(self) -> bool:
        """Validate JSONL format without loading all data."""
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    self._parse_example(data, line_num)
                    
                    # Only validate first few lines for performance
                    if line_num >= 10:
                        break
                        
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
