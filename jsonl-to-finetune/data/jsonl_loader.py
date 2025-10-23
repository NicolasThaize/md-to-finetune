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
        
        # Support multiple formats: "question/answer", "user/assistant", and "messages"
        if "question" in data and "answer" in data:
            question = data["question"]
            answer = data["answer"]
        elif "user" in data and "assistant" in data:
            question = data["user"]
            answer = data["assistant"]
        elif "messages" in data:
            # Handle messages format with role/content
            messages = data["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_num}: messages must be a list with at least 2 items")
            # Validate Harmony usage strictly; raises if invalid
            self._validate_harmony_messages(messages, line_num)
            
            # Find user and assistant messages
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(f"Line {line_num}: Each message must have 'role' and 'content' fields")
                
                if msg["role"] == "user" and user_msg is None:
                    user_msg = msg["content"]
                elif msg["role"] == "assistant" and assistant_msg is None:
                    # Prefer assistant content on channel "final" if present; otherwise take first
                    if msg.get("channel") in (None, "final"):
                        assistant_msg = msg["content"]
            
            if user_msg is None or assistant_msg is None:
                raise ValueError(f"Line {line_num}: messages must contain both 'user' and 'assistant' roles")
            
            question = user_msg
            answer = assistant_msg
        else:
            raise ValueError(f"Line {line_num}: Missing required fields. Expected 'question/answer', 'user/assistant', or 'messages'")
        
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"Line {line_num}: question and answer must be strings")
        
        if not question.strip() or not answer.strip():
            raise ValueError(f"Line {line_num}: question and answer cannot be empty")
        
        return JSONLTrainingExample(question.strip(), answer.strip())

    def _validate_harmony_messages(self, messages: List[Dict], line_num: int) -> None:
        """Validate messages according to OpenAI Harmony usage.
        See: https://cookbook.openai.com/articles/openai-harmony
        """
        allowed_roles = {"system", "developer", "user", "assistant"}
        allowed_channels = {"final", "analysis", "commentary"}

        seen_user = False
        seen_assistant = False

        for idx, msg in enumerate(messages, start=1):
            if not isinstance(msg, dict):
                raise ValueError(f"Line {line_num}: message[{idx}] must be an object")

            role = msg.get("role")
            content = msg.get("content")
            channel = msg.get("channel")

            if not isinstance(role, str) or not role:
                raise ValueError(f"Line {line_num}: message[{idx}] missing valid 'role'")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Line {line_num}: message[{idx}] missing non-empty 'content'")

            # Channel (if present) must be one of the Harmony channels
            if channel is not None and channel not in allowed_channels:
                raise ValueError(f"Line {line_num}: message[{idx}] has invalid channel '{channel}' (allowed: {sorted(allowed_channels)})")

            # Tool messages: role can be any tool name (not in base roles)
            is_tool_message = role not in allowed_roles
            if is_tool_message:
                # Must target assistant and be on commentary channel per Harmony
                to_field = msg.get("to") or msg.get("recipient")
                if to_field not in ("assistant", "functions", None):
                    raise ValueError(f"Line {line_num}: message[{idx}] tool message must specify 'to=assistant' or valid recipient")
                if channel not in (None, "commentary"):
                    raise ValueError(f"Line {line_num}: message[{idx}] tool message must use 'commentary' channel if channel is set")

            # Track presence of at least one user and assistant (final)
            if role == "user":
                seen_user = True
            if role == "assistant" and (channel in (None, "final")):
                seen_assistant = True

        # Require at least one user and one assistant message for a usable training pair
        if not seen_user:
            raise ValueError(f"Line {line_num}: Harmony messages must include at least one 'user' message")
        if not seen_assistant:
            raise ValueError(f"Line {line_num}: Harmony messages must include at least one 'assistant' message on channel 'final' (or no channel)")
    
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
