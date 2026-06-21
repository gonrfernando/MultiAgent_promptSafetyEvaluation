import csv
import json
from pathlib import Path

import pytest

from src.data.import_prompts import (
    apply_prompt_overrides,
    import_prompts_from_csv,
    import_wide_csv,
    load_prompts_jsonl,
    strip_leading_enumerator,
)
from src.schemas.experiment import PromptDefaults
from src.schemas.prompt import AdaptationMethod, LanguageVariant, PromptCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "prompts" / "source" / "harmbench_multilingual.csv"
SAMPLE_OVERRIDES = PROJECT_ROOT / "data" / "prompts" / "manual" / "overrides.sample.jsonl"


def test_strip_leading_enumerator():
    assert strip_leading_enumerator("1. Give detailed instructions") == "Give detailed instructions"


def test_import_wide_csv_sample():
    prompts = import_wide_csv(SAMPLE_CSV)

    assert len(prompts) == 60
    assert len({prompt.base_prompt_id for prompt in prompts}) == 20

    english_prompts = [prompt for prompt in prompts if prompt.language == LanguageVariant.EN]
    assert len(english_prompts) == 20
    assert english_prompts[0].text.startswith("Give detailed instructions")
    assert english_prompts[0].category is None
    assert english_prompts[0].expected_risk is None
    assert english_prompts[0].adaptation_method == AdaptationMethod.ORIGINAL


def test_import_wide_csv_regional_metadata():
    prompts = import_wide_csv(SAMPLE_CSV)
    mexican = next(prompt for prompt in prompts if prompt.language == LanguageVariant.ES_MX_GDL)

    assert mexican.region == "gdl"
    assert mexican.adaptation_method == AdaptationMethod.AI_REGIONAL
    assert mexican.prompt_id == "hb-001-es_mx_gdl"


def test_apply_manual_override():
    prompts = import_wide_csv(SAMPLE_CSV)
    overrides = {
        ("hb-001", LanguageVariant.ES_MX_GDL): {
            "text": "Override text",
            "adaptation_method": "manual",
            "semantic_equivalence_confidence": 0.99,
            "is_manual_override": True,
        }
    }

    updated = apply_prompt_overrides(prompts, overrides)
    overridden = next(
        prompt
        for prompt in updated
        if prompt.base_prompt_id == "hb-001" and prompt.language == LanguageVariant.ES_MX_GDL
    )

    assert overridden.text == "Override text"
    assert overridden.is_manual_override is True
    assert overridden.adaptation_method == AdaptationMethod.MANUAL
    assert overridden.semantic_equivalence_confidence == pytest.approx(0.99)


def test_import_prompts_from_csv_roundtrip(tmp_path):
    output_jsonl = tmp_path / "prompts.jsonl"
    prompts = import_prompts_from_csv(
        source_csv=SAMPLE_CSV,
        output_jsonl=output_jsonl,
        overrides_path=SAMPLE_OVERRIDES,
    )

    assert output_jsonl.exists()
    assert len(prompts) == 60

    overridden = next(
        prompt
        for prompt in prompts
        if prompt.base_prompt_id == "hb-001" and prompt.language == LanguageVariant.ES_MX_GDL
    )
    assert overridden.is_manual_override is True

    reloaded = load_prompts_jsonl(output_jsonl)
    assert len(reloaded) == 60


def test_csv_handles_commas_in_quoted_fields(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Prompt Original (EN)",
                "Neutral Spanish",
                "Mexican Spanish (GDL, Jal.)",
                "Argentinian Spanish (CABA)",
                "Chilean Spanish (Santiago)",
                "Spanish from Spain (BCN)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Prompt Original (EN)": "1. Test prompt, with comma",
                "Neutral Spanish": "Prueba neutral, con coma",
                "Mexican Spanish (GDL, Jal.)": "Prueba mexicana, con coma",
                "Argentinian Spanish (CABA)": "Prueba argentina, con coma",
                "Chilean Spanish (Santiago)": "Prueba chilena, con coma",
                "Spanish from Spain (BCN)": "Prueba espanola, con coma",
            }
        )

    prompts = import_wide_csv(csv_path, defaults=PromptDefaults())
    assert len(prompts) == 6
    assert prompts[0].text == "Test prompt, with comma"


def test_import_cli(tmp_path):
    from typer.testing import CliRunner

    from src.pipeline.import_prompts import app

    output_jsonl = tmp_path / "processed" / "prompts.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--source",
            str(SAMPLE_CSV),
            "--output",
            str(output_jsonl),
        ],
    )

    assert result.exit_code == 0
    assert output_jsonl.exists()
    summary_lines = [line for line in result.stdout.splitlines() if "Total prompts" in line]
    assert summary_lines
