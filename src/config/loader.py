from pathlib import Path

import yaml

from src.schemas.experiment import ExperimentConfig, PromptDefaults
from src.schemas.prompt import LanguageVariant, PromptCategory


def load_experiment_config(config_path: Path | str) -> ExperimentConfig:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    experiment = raw["experiment"]
    paths = raw["paths"]
    generation = raw.get("generation", {})
    evaluation = raw.get("evaluation", {})
    prompt_defaults_raw = raw.get("prompt_defaults", {})

    category_raw = prompt_defaults_raw.get("category")
    category = PromptCategory(category_raw) if category_raw else None
    expected_risk = prompt_defaults_raw.get("expected_risk")

    judges = raw.get("judges", raw.get("evaluators", ["security", "policy", "critical"]))
    evaluators = raw.get("evaluators", judges)

    return ExperimentConfig(
        name=experiment["name"],
        version=experiment["version"],
        seed=experiment.get("seed", 42),
        prompts_dir=paths["prompts_dir"],
        prompts_file=paths.get("prompts_file", "data/prompts/processed/prompts.jsonl"),
        prompts_source_txt=paths.get("prompts_source_txt", "prompts.txt"),
        prompts_source_csv=paths.get("prompts_source_csv", "data/prompts/source/harmbench_multilingual.csv"),
        prompts_overrides=paths.get("prompts_overrides", "data/prompts/manual/overrides.jsonl"),
        judges_config=paths.get("judges_config", "configs/judges.yaml"),
        responses_dir=paths["responses_dir"],
        evaluations_dir=paths["evaluations_dir"],
        labels_dir=paths["labels_dir"],
        results_dir=paths["results_dir"],
        languages=[LanguageVariant(language) for language in raw["languages"]],
        judges=list(judges),
        evaluators=list(evaluators),
        prompt_defaults=PromptDefaults(
            source=prompt_defaults_raw.get("source", "harmbench"),
            category=category,
            expected_risk=expected_risk,
        ),
        generation_temperature=generation.get("temperature", 0.7),
        generation_max_tokens=generation.get("max_tokens", 1024),
        evaluation_temperature=evaluation.get("temperature", 0.0),
        evaluation_max_tokens=evaluation.get("max_tokens", 512),
    )
