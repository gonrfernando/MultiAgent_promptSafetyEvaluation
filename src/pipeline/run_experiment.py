import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from src.config.loader import load_experiment_config
from src.disagreement.metrics import compute_disagreement_metrics
from src.pipeline.mock_data import (
    build_mock_evaluations,
    build_mock_prompts,
    build_mock_responses,
    snapshot_config,
    write_jsonl,
)
from src.schemas.experiment import ExperimentRun

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Run with mock data and write JSONL outputs."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Path to experiment YAML config."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run identifier."),
) -> None:
    config_path = config if config.is_absolute() else PROJECT_ROOT / config
    experiment_config = load_experiment_config(config_path)
    resolved_run_id = run_id or f"dry-{uuid4().hex[:8]}"

    run = ExperimentRun(
        run_id=resolved_run_id,
        config=experiment_config,
        dry_run=dry_run,
    )

    results_dir = _resolve_path(experiment_config.results_dir) / resolved_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config_path, results_dir / "config_snapshot.yaml")

    if not dry_run:
        console.print("[yellow]Live mode is not implemented yet. Use --dry-run for Phase 0.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"[bold green]Starting dry run[/bold green] run_id={resolved_run_id}")

    prompts = build_mock_prompts()
    responses = build_mock_responses(
        run_id=resolved_run_id,
        prompts=prompts,
        model="mock-target-model",
        temperature=experiment_config.generation_temperature,
        max_tokens=experiment_config.generation_max_tokens,
    )
    evaluations = build_mock_evaluations(responses, model="mock-evaluator-model")

    disagreement_records = []
    for response in responses:
        response_evaluations = [item for item in evaluations if item.response_id == response.response_id]
        disagreement_records.append(
            compute_disagreement_metrics(response.response_id, response_evaluations)
        )

    write_jsonl(results_dir / "prompts.jsonl", prompts)
    write_jsonl(results_dir / "responses.jsonl", responses)
    write_jsonl(results_dir / "evaluations.jsonl", evaluations)
    write_jsonl(results_dir / "disagreement_metrics.jsonl", disagreement_records)

    run.prompt_count = len(prompts)
    run.response_count = len(responses)
    run.evaluation_count = len(evaluations)
    run.finished_at = datetime.now(timezone.utc)
    run.notes = "Phase 0 dry run with mock prompts, responses, and evaluator outputs."

    with (results_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)

    summary = {
        "run_id": resolved_run_id,
        "prompt_count": run.prompt_count,
        "response_count": run.response_count,
        "evaluation_count": run.evaluation_count,
        "disagreement_records": len(disagreement_records),
        "results_dir": str(results_dir),
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    console.print("[bold green]Dry run complete[/bold green]")
    console.print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
