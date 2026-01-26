from pathlib import Path
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "cbm-concept-inference"
    env: str = "production"
    device: str = "auto"
    img_size: int = 320
    alpha_overlay: float = 0.45

    # ✅ Pydantic v2 style config (removes warning)
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def configs_dir(self) -> Path:
        return PROJECT_ROOT / "configs"

    @property
    def backbone_path(self) -> Path:
        return self.models_dir / "convnext" / "timmconvnext_base.fb_in22k_ft_in1k.safetensors"

    @property
    def last_stage_path(self) -> Path:
        return self.models_dir / "convnext" / "convnext_base_last_stage_finetuned.pth"

    @property
    def concept_head_path(self) -> Path:
        return self.models_dir / "base_backbone_concept_head.pth"

    @property
    def thresholds_path(self) -> Path:
        return self.configs_dir / "thresholds.json"

    @property
    def concepts_path(self) -> Path:
        return self.configs_dir / "concepts.yaml"

    @property
    def device_torch(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = Settings()
