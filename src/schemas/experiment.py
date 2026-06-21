from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.schemas.prompt import LanguageVariant, PromptCategory


class PromptDefaults(BaseModel):
    source: str = "harmbench"
    category: PromptCategory | None = None
    expected_risk: float | None = Field(default=None, ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    name: str
    version: str
    seed: int = 42
    prompts_dir: str
    prompts_file: str = "data/prompts/processed/prompts.jsonl"
    prompts_source_txt: str = "prompts.txt"
    prompts_source_csv: str = "data/prompts/source/harmbench_multilingual.csv"
    prompts_overrides: str = "data/prompts/manual/overrides.jsonl"
    judges_config: str = "configs/judges.yaml"
    responses_dir: str
    evaluations_dir: str
    labels_dir: str
    results_dir: str
    languages: list[LanguageVariant]
    judges: list[str] = Field(default_factory=lambda: ["security", "policy", "critical"])
    evaluators: list[str] = Field(default_factory=list)
    prompt_defaults: PromptDefaults = Field(default_factory=PromptDefaults)
    generation_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1024, ge=1)
    evaluation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    evaluation_max_tokens: int = Field(default=512, ge=1)


class ExperimentRun(BaseModel):
    run_id: str
    config: ExperimentConfig
    dry_run: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    prompt_count: int = 0
    response_count: int = 0
    evaluation_count: int = 0
    notes: str | None = None
