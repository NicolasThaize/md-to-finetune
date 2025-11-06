from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


class DataManager:
    """Handles input TXT files and exposes helpers to read them.

    Data layer abstraction.
    """

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def add_txt_file(self, file_path: str | Path) -> Path:
        src = Path(file_path)
        if not src.exists() or src.suffix.lower() != ".txt":
            raise ValueError("Le fichier doit exister et être au format .txt")
        dst = self.input_dir / src.name
        if src.resolve() != dst.resolve():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return dst

    def list_txt_files(self) -> List[Path]:
        return sorted(self.input_dir.glob("*.txt"))

    def read_lines(self, path: Path) -> Iterable[str]:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            yield line


