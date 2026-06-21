import pytest

from src.data.import_prompts_txt import import_prompts_txt
from src.evaluation.judge import build_user_message, parse_judge_response
from src.schemas.evaluation import JudgeRole
from src.schemas.prompt import AdaptationMethod, LanguageVariant, Prompt

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
PROMPTS_TXT = PROJECT_ROOT / "prompts.txt"


def test_import_prompts_txt_real_file():
    prompts = import_prompts_txt(PROMPTS_TXT)

    assert len(prompts) == 120
    assert prompts[0].language == LanguageVariant.ES_NEUTRAL
    assert prompts[1].language == LanguageVariant.ES_MX_GDL
    assert prompts[5].language == LanguageVariant.ES_UY
    assert prompts[0].base_prompt_id == "hb-001"
    assert prompts[6].base_prompt_id == "hb-002"
    assert prompts[0].category is None
    assert prompts[0].expected_risk is None


def test_import_prompts_txt_block_order(tmp_path):
    lines = [f"prompt-{index}" for index in range(12)]
    source = tmp_path / "sample.txt"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prompts = import_prompts_txt(source)

    assert len(prompts) == 12
    assert [prompt.language for prompt in prompts[:6]] == [
        LanguageVariant.ES_NEUTRAL,
        LanguageVariant.ES_MX_GDL,
        LanguageVariant.ES_AR_CABA,
        LanguageVariant.ES_CL_SANTIAGO,
        LanguageVariant.ES_ES_BCN,
        LanguageVariant.ES_UY,
    ]
    assert prompts[6].base_prompt_id == "hb-002"


def test_build_user_message_only_contains_prompt_text():
    prompt = Prompt(
        prompt_id="hb-001-es_neutral",
        base_prompt_id="hb-001",
        language=LanguageVariant.ES_NEUTRAL,
        text="Texto de prueba",
        adaptation_method=AdaptationMethod.AI_NEUTRAL,
    )
    message = build_user_message(prompt)

    assert "Texto de prueba" in message
    assert "expected_risk" not in message
    assert "category" not in message


def test_parse_judge_response_success():
    prompt = Prompt(
        prompt_id="hb-001-es_neutral",
        base_prompt_id="hb-001",
        language=LanguageVariant.ES_NEUTRAL,
        text="Texto de prueba",
    )
    evaluation = parse_judge_response(
        content='{"safe": false, "risk_score": 0.91, "explanation": "Harmful", "risk_category": "weapons"}',
        prompt=prompt,
        judge_id="security",
        role=JudgeRole.SECURITY,
        model="llama3.1:8b",
        run_id="test-run",
        evaluation_id="eval-1",
    )

    assert evaluation.safe is False
    assert evaluation.risk_score == pytest.approx(0.91)
    assert evaluation.judge_refused is False


def test_parse_judge_response_failure():
    prompt = Prompt(
        prompt_id="hb-001-es_neutral",
        base_prompt_id="hb-001",
        language=LanguageVariant.ES_NEUTRAL,
        text="Texto de prueba",
    )
    evaluation = parse_judge_response(
        content="Lo siento, no puedo ayudarte.",
        prompt=prompt,
        judge_id="security",
        role=JudgeRole.SECURITY,
        model="llama3.1:8b",
        run_id="test-run",
        evaluation_id="eval-1",
    )

    assert evaluation.judge_refused is True
    assert evaluation.parse_error is not None
