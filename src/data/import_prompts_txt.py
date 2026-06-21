from __future__ import annotations

import json
from pathlib import Path

from src.data.txt_variants import TXT_VARIANT_ORDER, VARIANTS_PER_BLOCK
from src.schemas.prompt import Prompt


def make_prompt_id(base_prompt_id: str, language) -> str:
    return f"{base_prompt_id}-{language.value}"


def make_base_prompt_id(source_index: int, prefix: str = "hb") -> str:
    return f"{prefix}-{source_index:03d}"


def import_prompts_txt(
    txt_path: Path | str,
    output_jsonl: Path | str | None = None,
    source_prefix: str = "hb",
) -> list[Prompt]:
    path = Path(txt_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        raise ValueError(f"No prompts found in {path}")

    if len(lines) % VARIANTS_PER_BLOCK != 0:
        raise ValueError(
            f"Expected a multiple of {VARIANTS_PER_BLOCK} non-empty lines in {path}, got {len(lines)}."
        )

    prompts: list[Prompt] = []
    for line_index, text in enumerate(lines):
        block_index = line_index // VARIANTS_PER_BLOCK
        variant_index = line_index % VARIANTS_PER_BLOCK
        source_index = block_index + 1
        base_prompt_id = make_base_prompt_id(source_index, prefix=source_prefix)
        variant = TXT_VARIANT_ORDER[variant_index]

        prompts.append(
            Prompt(
                prompt_id=make_prompt_id(base_prompt_id, variant.language),
                base_prompt_id=base_prompt_id,
                language=variant.language,
                text=text,
                source="harmbench",
                source_index=source_index,
                adaptation_method=variant.adaptation_method,
                region=variant.region,
            )
        )

    if output_jsonl is not None:
        write_prompts_jsonl(output_jsonl, prompts)

    return prompts


def write_prompts_jsonl(path: Path | str, prompts: list[Prompt]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n")


def load_prompts_jsonl(path: Path | str) -> list[Prompt]:
    from src.schemas.prompt import Prompt

    input_path = Path(path)
    prompts: list[Prompt] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            prompts.append(Prompt.model_validate(json.loads(cleaned)))
    return prompts
