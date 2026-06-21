from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.progress import Progress

from src.config.judges_loader import load_judges_config
from src.config.loader import load_experiment_config
from src.data.import_prompts import load_prompts_jsonl
from src.disagreement.metrics import compute_disagreement_metrics
from src.evaluation.judge import evaluate_prompt_with_judge
from src.schemas.evaluation import PromptEvaluation

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_JUDGES = PROJECT_ROOT / "configs" / "judges.yaml"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _evaluation_key(evaluation: PromptEvaluation) -> tuple[str, str]:
    return evaluation.prompt_id, evaluation.judge_id


def load_existing_evaluations(path: Path) -> dict[tuple[str, str], PromptEvaluation]:
    if not path.exists():
        return {}

    existing: dict[tuple[str, str], PromptEvaluation] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            evaluation = PromptEvaluation.model_validate(json.loads(cleaned))
            existing[_evaluation_key(evaluation)] = evaluation
    return existing


def append_evaluation(path: Path, evaluation: PromptEvaluation) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False) + "\n")


@app.command()
def main(
    run_id: str | None = typer.Option(None, "--run-id", help="Evaluation run identifier."),
    limit: int | None = typer.Option(None, "--limit", help="Evaluate only the first N prompts."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Experiment config YAML."),
    judges_config: Path = typer.Option(DEFAULT_JUDGES, "--judges-config", help="Judges config YAML."),
    prompts_file: Path | None = typer.Option(None, "--prompts-file", help="Prompts JSONL input."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip existing prompt/judge pairs."),
) -> None:
    experiment_config = load_experiment_config(_resolve_path(config))
    judges = load_judges_config(_resolve_path(judges_config))
    prompts_path = _resolve_path(prompts_file or Path(experiment_config.prompts_file))

    if not prompts_path.exists():
        console.print(f"[red]Prompts file not found:[/red] {prompts_path}")
        console.print("Run: python -m src.pipeline.import_prompts_txt")
        raise typer.Exit(code=1)

    resolved_run_id = run_id or f"judge-{uuid4().hex[:8]}"
    run_dir = _resolve_path(Path(experiment_config.evaluations_dir)) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluations_path = run_dir / "evaluations.jsonl"

    prompts = load_prompts_jsonl(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    existing = load_existing_evaluations(evaluations_path) if resume else {}
    total_tasks = len(prompts) * len(judges.judges)
    completed = 0
    skipped = 0
    refused = 0

    console.print(f"[bold green]Starting prompt evaluation[/bold green] run_id={resolved_run_id}")
    console.print(f"Prompts: {len(prompts)} | Judges: {len(judges.judges)} | Tasks: {total_tasks}")

    with Progress() as progress:
        task_id = progress.add_task("Evaluating prompts", total=total_tasks)

        for judge in judges.judges:
            console.print(f"Judge [bold]{judge.id}[/bold] model={judge.model}")
            for prompt in prompts:
                key = (prompt.prompt_id, judge.id)
                if resume and key in existing:
                    skipped += 1
                    progress.advance(task_id)
                    continue

                evaluation_id = f"eval-{prompt.prompt_id}-{judge.id}"
                evaluation = evaluate_prompt_with_judge(
                    prompt=prompt,
                    judge_id=judge.id,
                    role=judge.role,
                    model=judge.model,
                    system_prompt=judge.system_prompt.strip(),
                    base_url=judges.ollama.base_url,
                    temperature=judges.ollama.temperature,
                    max_tokens=judges.ollama.max_tokens,
                    run_id=resolved_run_id,
                    evaluation_id=evaluation_id,
                )
                append_evaluation(evaluations_path, evaluation)
                existing[key] = evaluation
                completed += 1
                if evaluation.judge_refused:
                    refused += 1
                progress.advance(task_id)

    disagreement_records = []
    for prompt in prompts:
        prompt_evaluations = [
            existing[(prompt.prompt_id, judge.id)]
            for judge in judges.judges
            if (prompt.prompt_id, judge.id) in existing
        ]
        if len(prompt_evaluations) == len(judges.judges):
            disagreement_records.append(
                compute_disagreement_metrics(prompt.prompt_id, prompt_evaluations)
            )

    disagreement_path = run_dir / "disagreement_metrics.jsonl"
    with disagreement_path.open("w", encoding="utf-8") as handle:
        for record in disagreement_records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    metadata = {
        "run_id": resolved_run_id,
        "prompt_count": len(prompts),
        "judge_count": len(judges.judges),
        "completed_evaluations": completed,
        "skipped_evaluations": skipped,
        "refused_or_parse_failures": refused,
        "evaluations_path": str(evaluations_path),
        "disagreement_path": str(disagreement_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    console.print("[bold green]Evaluation complete[/bold green]")
    console.print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app()
