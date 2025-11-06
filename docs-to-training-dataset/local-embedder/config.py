"""Configuration et valeurs par défaut pour la pipeline."""

import logging
import os


DEFAULT_LLM_MODEL = os.getenv("LOCAL_EMBEDDER_LLM_MODEL", "mistral:7b-instruct")
DEFAULT_MISTRAL_TOKENIZER_MODEL = os.getenv(
    "LOCAL_EMBEDDER_MISTRAL_TOKENIZER_MODEL", "mistralai/Mistral-7B-v0.1"
)
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LOCAL_EMBEDDER_LLM_TEMPERATURE", "0.1"))
DEFAULT_LLM_TIMEOUT = float(os.getenv("LOCAL_EMBEDDER_LLM_TIMEOUT", "60.0"))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)

import logging
import os


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)


DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct")
DEFAULT_MISTRAL_TOKENIZER = os.getenv("MISTRAL_TOKENIZER", "mistralai/Mistral-7B-v0.1")
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "60.0"))


