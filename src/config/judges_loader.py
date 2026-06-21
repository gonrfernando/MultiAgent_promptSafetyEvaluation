from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.schemas.evaluation import JudgeRole


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


class JudgeConfig(BaseModel):
    id: str
    role: JudgeRole
    model: str
    system_prompt: str


class JudgesConfig(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    judges: list[JudgeConfig]


def load_judges_config(config_path: Path | str) -> JudgesConfig:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return JudgesConfig.model_validate(raw)
