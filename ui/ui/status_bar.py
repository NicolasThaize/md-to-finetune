import tkinter as tk
from typing import Protocol


class StatusObserver(Protocol):
    def on_status_update(self, message: str) -> None:
        ...


class StatusBar(tk.Frame):
    """A simple status bar that can be notified of status updates."""

    def __init__(self, master: tk.Misc | None = None) -> None:
        super().__init__(master)
        self._status_var = tk.StringVar(value="Prêt.")
        self._label = tk.Label(self, textvariable=self._status_var, anchor="w")
        self._label.pack(fill=tk.X)

    def set_status(self, message: str) -> None:
        self._status_var.set(message)


