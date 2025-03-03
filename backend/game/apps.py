from django.apps import AppConfig


class GameConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'game'

    # def ready(self):
    #     model_path = hf_hub_download(
    #         local_dir=".",
    #         repo_id="fairportrobotics/rock-paper-scissors",
    #         filename="model.pt"
    #     )
    #     self.model = YOLO(model_path)
