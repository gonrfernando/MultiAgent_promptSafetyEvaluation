from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from src.data.column_mapping import OPTIONAL_METADATA_COLUMNS, normalize_header, resolve_column_spec
from src.schemas.experiment import PromptDefaults
from src.schemas.prompt import AdaptationMethod, LanguageVariant, Prompt, PromptCategory


def strip_leading_enumerator(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text.strip())


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return float(cleaned)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return int(float(cleaned))


def make_prompt_id(base_prompt_id: str, language: LanguageVariant) -> str:
    return f"{base_prompt_id}-{language.value}"


def make_base_prompt_id(source_index: int, prefix: str = "hb") -> str:
    return f"{prefix}-{source_index:03d}"


def import_wide_csv(
    csv_path: Path | str,
    defaults: PromptDefaults | None = None,
    source_prefix: str = "hb",
) -> list[Prompt]:
    path = Path(csv_path)
    prompt_defaults = defaults or PromptDefaults()
    prompts: list[Prompt] = []

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        mapped_columns: dict[str, object] = {}
        metadata_columns: dict[str, str] = {}
        for header in reader.fieldnames:
            normalized = normalize_header(header)
            spec = resolve_column_spec(header)
            if spec is not None:
                mapped_columns[header] = spec
            elif normalized in OPTIONAL_METADATA_COLUMNS:
                metadata_columns[normalized] = header

        if not mapped_columns:
            raise ValueError(
                f"No recognized prompt columns found in {path}. "
                "Expected headers like 'Prompt Original (EN)', 'Neutral Spanish', etc."
            )

        for row_index, row in enumerate(reader, start=1):
            source_index = parse_optional_int(row.get(metadata_columns.get("source_index", ""), "")) or row_index
            base_prompt_id = make_base_prompt_id(source_index, prefix=source_prefix)
            harmbench_behavior = row.get(metadata_columns.get("harmbench_behavior", ""), "").strip() or None

            category_raw = row.get(metadata_columns.get("category", ""), "").strip()
            category = PromptCategory(category_raw) if category_raw else prompt_defaults.category

            expected_risk = parse_optional_float(row.get(metadata_columns.get("expected_risk", ""), ""))
            if expected_risk is None:
                expected_risk = prompt_defaults.expected_risk

            semantic_confidence = parse_optional_float(
                row.get(metadata_columns.get("semantic_equivalence_confidence", ""), "")
            )

            for header, spec in mapped_columns.items():
                raw_text = (row.get(header) or "").strip()
                if not raw_text:
                    continue

                text = strip_leading_enumerator(raw_text)
                prompts.append(
                    Prompt(
                        prompt_id=make_prompt_id(base_prompt_id, spec.language),
                        base_prompt_id=base_prompt_id,
                        language=spec.language,
                        text=text,
                        category=category,
                        expected_risk=expected_risk,
                        source=prompt_defaults.source,
                        source_index=source_index,
                        harmbench_behavior=harmbench_behavior,
                        adaptation_method=spec.adaptation_method,
                        region=spec.region,
                        semantic_equivalence_confidence=semantic_confidence,
                        is_manual_override=False,
                    )
                )

    return prompts


def load_prompt_overrides(path: Path | str) -> dict[tuple[str, LanguageVariant], dict]:
    override_path = Path(path)
    if not override_path.exists():
        return {}

    overrides: dict[tuple[str, LanguageVariant], dict] = {}
    with override_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            payload = json.loads(cleaned)
            base_prompt_id = payload["base_prompt_id"]
            language = LanguageVariant(payload["language"])
            overrides[(base_prompt_id, language)] = payload

    return overrides


def apply_prompt_overrides(prompts: list[Prompt], overrides: dict[tuple[str, LanguageVariant], dict]) -> list[Prompt]:
    if not overrides:
        return prompts

    updated: list[Prompt] = []
    for prompt in prompts:
        key = (prompt.base_prompt_id, prompt.language)
        if key not in overrides:
            updated.append(prompt)
            continue

        override = overrides[key]
        updated.append(
            prompt.model_copy(
                update={
                    "text": override.get("text", prompt.text),
                    "adaptation_method": AdaptationMethod(
                        override.get("adaptation_method", AdaptationMethod.MANUAL.value)
                    ),
                    "semantic_equivalence_confidence": override.get(
                        "semantic_equivalence_confidence",
                        prompt.semantic_equivalence_confidence,
                    ),
                    "is_manual_override": override.get("is_manual_override", True),
                    "category": PromptCategory(override["category"]) if override.get("category") else prompt.category,
                    "expected_risk": override.get("expected_risk", prompt.expected_risk),
                    "region": override.get("region", prompt.region),
                }
            )
        )

    return updated


def write_prompts_jsonl(path: Path | str, prompts: list[Prompt]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n")


def load_prompts_jsonl(path: Path | str) -> list[Prompt]:
    input_path = Path(path)
    prompts: list[Prompt] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            prompts.append(Prompt.model_validate(json.loads(cleaned)))
    return prompts


def import_prompts_from_csv(
    source_csv: Path | str,
    output_jsonl: Path | str,
    overrides_path: Path | str | None = None,
    defaults: PromptDefaults | None = None,
    source_prefix: str = "hb",
) -> list[Prompt]:
    prompts = import_wide_csv(source_csv, defaults=defaults, source_prefix=source_prefix)
    overrides = load_prompt_overrides(overrides_path) if overrides_path else {}
    prompts = apply_prompt_overrides(prompts, overrides)
    write_prompts_jsonl(output_jsonl, prompts)
    return prompts
