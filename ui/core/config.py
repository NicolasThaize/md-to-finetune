from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data_input: Path
    data_output: Path
    logs: Path


def get_default_paths() -> AppPaths:
    root = Path(__file__).resolve().parents[1]
    return AppPaths(
        root=root,
        data_input=root / "data" / "input",
        data_output=root / "data" / "output",
        logs=root / "logs",
    )


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "trained-model"
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 8


