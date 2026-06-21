from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.config.loader import load_experiment_config
from src.schemas.evaluation import AgentEvaluation, EvaluatorRole
from src.schemas.experiment import ExperimentConfig, ExperimentRun
from src.schemas.prompt import LanguageVariant, Prompt, PromptCategory
from src.schemas.response import GeneratedResponse


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def test_load_experiment_config():
    config = load_experiment_config(CONFIG_PATH)

    assert config.name == "agent-disagreement-safety"
    assert config.version == "0.1.0"
    assert config.seed == 42
    assert LanguageVariant.ES_NEUTRAL in config.languages
    assert LanguageVariant.ES_UY in config.languages
    assert "security" in config.evaluators
    assert config.prompts_file.endswith("prompts.jsonl")
    assert config.prompts_source_txt == "prompts.txt"
    assert config.judges == ["security", "policy", "critical"]


def test_prompt_schema_roundtrip():
    prompt = Prompt(
        prompt_id="prompt-001",
        base_prompt_id="bp-001",
        language=LanguageVariant.ES_MX_GDL,
        text="Hola",
        category=PromptCategory.BENIGN,
        expected_risk=0.1,
    )

    restored = Prompt.model_validate(prompt.model_dump())
    assert restored.prompt_id == "prompt-001"
    assert restored.language == LanguageVariant.ES_MX_GDL


def test_generated_response_schema():
    response = GeneratedResponse(
        response_id="resp-001",
        prompt_id="prompt-001",
        base_prompt_id="bp-001",
        language=LanguageVariant.EN,
        model="mock-model",
        text="Example response",
        run_id="run-001",
        generation_temperature=0.7,
        generation_max_tokens=1024,
    )

    assert response.run_id == "run-001"


def test_agent_evaluation_risk_bounds():
    evaluation = AgentEvaluation(
        evaluation_id="eval-001",
        response_id="resp-001",
        agent_id="security",
        role=EvaluatorRole.SECURITY,
        safe=True,
        risk_score=0.2,
        explanation="Looks safe.",
        category="benign",
        model="mock-model",
    )

    assert evaluation.safe is True

    with pytest.raises(ValidationError):
        AgentEvaluation(
            evaluation_id="eval-002",
            response_id="resp-001",
            agent_id="security",
            role=EvaluatorRole.SECURITY,
            safe=True,
            risk_score=1.5,
            explanation="Invalid score.",
            category="benign",
            model="mock-model",
        )


def test_experiment_run_defaults():
    config = ExperimentConfig(
        name="test",
        version="0.0.1",
        prompts_dir="data/prompts",
        responses_dir="data/responses",
        evaluations_dir="data/evaluations",
        labels_dir="data/labels",
        results_dir="results/runs",
        languages=[LanguageVariant.EN],
        evaluators=["security"],
        judges=["security"],
    )
    run = ExperimentRun(run_id="run-test", config=config, dry_run=True)

    assert run.finished_at is None
    assert run.prompt_count == 0
    assert isinstance(run.started_at, datetime)
