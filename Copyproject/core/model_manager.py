from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Protocol

from transformers import AutoModelForCausalLM, AutoTokenizer


# Observer pattern
class ModelObserver(Protocol):
    def on_model_changed(self, model_name: str) -> None:
        ...

    def on_status(self, message: str) -> None:
        ...


# Factory pattern for model creation (stubbed for now)
class PretrainedModel(Protocol):
    name: str
    local_dir: Path


@dataclass
class SimpleModel:
    name: str
    local_dir: Path


class ModelFactory:
    registry: Dict[str, Callable[[Path], PretrainedModel]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[[Path], PretrainedModel]) -> None:
        cls.registry[name] = builder

    @classmethod
    def create(cls, name: str, output_dir: Path) -> PretrainedModel:
        if name not in cls.registry:
            # Default to a simple placeholder model
            return SimpleModel(name=name, local_dir=output_dir / name.replace('/', '__'))
        return cls.registry[name](output_dir)


class ModelManager:
    """Business logic to manage pretrained models using a Factory and notifying Observers."""

    def __init__(self, models_output_dir: Path) -> None:
        self._current_model: PretrainedModel | None = None
        self._observers: List[ModelObserver] = []
        self._models_output_dir = models_output_dir
        self._models_output_dir.mkdir(parents=True, exist_ok=True)

    def add_observer(self, observer: ModelObserver) -> None:
        self._observers.append(observer)

    def _notify_status(self, message: str) -> None:
        for obs in self._observers:
            obs.on_status(message)

    def _notify_model_changed(self, model_name: str) -> None:
        for obs in self._observers:
            obs.on_model_changed(model_name)

    def select_model(self, model_name: str) -> None:
        self._notify_status(f"Chargement du modèle '{model_name}' (depuis le cache ou en ligne)...")
        self._current_model = ModelFactory.create(model_name, self._models_output_dir)
        self._notify_model_changed(self._current_model.name)
        self._notify_status("Modèle prêt.")

    @property
    def current_model_name(self) -> str | None:
        return self._current_model.name if self._current_model else None


# --- Concrete HF builder registrations ---

def _build_gpt_oss_20b(output_root: Path) -> PretrainedModel:
    model_id = "openai/gpt-oss-20b"
    local_dir = output_root / model_id.replace('/', '__')
    local_dir.mkdir(parents=True, exist_ok=True)

    try: 
        # Load from cache if available; transformers will fallback to online otherwise
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Persist to the designated output folder
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

        return SimpleModel(name=model_id, local_dir=local_dir)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            raise Exception(f"Accès refusé au modèle {model_id}. Veuillez vérifier vos permissions ou utilisez un modèle disponible en ligne.")
        raise e # re-raise the exception if it's not a 401 or gated error


# Register available models in the factory
ModelFactory.register("openai/gpt-oss-20b", _build_gpt_oss_20b)


