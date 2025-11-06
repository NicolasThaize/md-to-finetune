"""Application entry point.

Initializes the Tkinter UI and wires minimal MVC placeholders.
"""

from pathlib import Path

from ui.main_window import MainWindow
from core.config import get_default_paths
from core.model_manager import ModelManager, ModelObserver
from core.data_manager import DataManager


class Controller(ModelObserver):
    def __init__(self, view: MainWindow) -> None:
        self.view = view
        self.paths = get_default_paths()
        self.models = ModelManager(models_output_dir=self.paths.data_output)
        self.models.add_observer(self)
        self.data = DataManager(input_dir=self.paths.data_input)

    # Observer callbacks
    def on_status(self, message: str) -> None:  # type: ignore[override]
        self.view.set_status(message)

    def on_model_changed(self, model_name: str) -> None:  # type: ignore[override]
        self.view.set_selected_model_name(model_name)

    # UI bound actions
    def select_gpt_oss_20b(self) -> None:
        # Only one selectable for now; uses cache or downloads, then saves to output
        self.models.select_model("openai/gpt-oss-20b")

    def upload_txt(self, path: str) -> None:
        saved = self.data.add_txt_file(path)
        self.view.set_status(f"Fichier importé: {saved.name}")

    def start_training(self) -> None:
        # Stub for now
        self.view.set_status("Pipeline de fine-tuning (stub)")


def main() -> None:
    app = MainWindow()
    controller = Controller(app)
    # Wire UI to controller
    app.on_select_model = controller.select_gpt_oss_20b
    app.on_upload_txt = controller.upload_txt
    app.on_start_training = controller.start_training
    app.set_status("Sélectionnez le modèle GPT-OSS-20B (cache ou en ligne)")
    app.mainloop()


if __name__ == "__main__":
    main()


