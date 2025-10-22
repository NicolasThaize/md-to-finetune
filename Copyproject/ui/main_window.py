import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable

from ui.status_bar import StatusBar


class MainWindow(tk.Tk):
    """Main application window (View in MVC)."""

    def __init__(self) -> None:
        super().__init__()
        self.title("LLM Fine-Tuning Tool")
        self.geometry("800x500")

        # Controller callbacks (to be injected later)
        self.on_select_model: Callable[[], None] | None = None
        self.on_upload_txt: Callable[[str], None] | None = None
        self.on_start_training: Callable[[], None] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=12, pady=12)

        self._model_label_var = tk.StringVar(value="Modèle: aucun sélectionné")
        model_label = tk.Label(btn_frame, textvariable=self._model_label_var, anchor="w")
        model_label.grid(row=0, column=0, padx=(0, 8), sticky="w")

        select_model_btn = tk.Button(btn_frame, text="Choisir un modèle", command=self._handle_select_model)
        select_model_btn.grid(row=0, column=1, padx=8)

        upload_btn = tk.Button(btn_frame, text="Uploader fichier TXT", command=self._handle_upload_txt)
        upload_btn.grid(row=0, column=2, padx=8)

        train_btn = tk.Button(btn_frame, text="Lancer le fine-tuning", command=self._handle_start_training)
        train_btn.grid(row=0, column=3, padx=8)

        # Expandable center area reserved for future widgets/log console
        self._center = tk.Frame(self)
        self._center.pack(expand=True, fill=tk.BOTH)

        # Status bar at the bottom
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # Public methods used by controller
    def set_selected_model_name(self, name: str) -> None:
        self._model_label_var.set(f"Modèle: {name}")

    def set_status(self, message: str) -> None:
        self.status_bar.set_status(message)

    # Event handlers
    def _handle_select_model(self) -> None:
        if self.on_select_model:
            self.on_select_model()
        else:
            messagebox.showinfo("Info", "Sélection de modèle non configurée (stub).")

    def _handle_upload_txt(self) -> None:
        file_path = filedialog.askopenfilename(title="Choisir un fichier TXT", filetypes=[("Text files", "*.txt")])
        if file_path and self.on_upload_txt:
            self.on_upload_txt(file_path)
        elif not self.on_upload_txt:
            messagebox.showinfo("Info", "Upload non configuré (stub).")

    def _handle_start_training(self) -> None:
        if self.on_start_training:
            self.on_start_training()
        else:
            messagebox.showinfo("Info", "Fine-tuning non configuré (stub).")


