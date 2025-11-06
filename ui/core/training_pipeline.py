from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class TrainingObserver(Protocol):
    def on_training_status(self, message: str) -> None:
        ...


class TrainingStrategy(ABC):
    """Strategy interface for various fine-tuning approaches."""

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class DummyTrainingStrategy(TrainingStrategy):
    """Placeholder training strategy (no-op)."""

    def run(self) -> None:
        # In the real implementation, run HF Trainer / PEFT / etc.
        pass


class TrainingPipeline:
    """Coordinates training with a chosen strategy and notifies observers."""

    def __init__(self, strategy: TrainingStrategy | None = None) -> None:
        self._strategy = strategy or DummyTrainingStrategy()
        self._observers: list[TrainingObserver] = []

    def set_strategy(self, strategy: TrainingStrategy) -> None:
        self._strategy = strategy

    def add_observer(self, observer: TrainingObserver) -> None:
        self._observers.append(observer)

    def _notify(self, message: str) -> None:
        for obs in self._observers:
            obs.on_training_status(message)

    def start(self) -> None:
        self._notify("Démarrage du fine-tuning...")
        self._strategy.run()
        self._notify("Fine-tuning terminé.")


