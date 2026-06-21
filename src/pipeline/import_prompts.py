from pathlib import Path

import typer
from rich.console import Console

from src.config.loader import load_experiment_config
from src.data.import_prompts import import_prompts_from_csv

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@app.command()
def main(
    source: Path | None = typer.Option(None, "--source", help="Wide-format HarmBench CSV."),
    output: Path | None = typer.Option(None, "--output", help="Processed prompts JSONL output."),
    overrides: Path | None = typer.Option(None, "--overrides", help="Manual overrides JSONL."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Experiment config YAML."),
) -> None:
    config_path = _resolve_path(config)
    experiment_config = load_experiment_config(config_path)

    source_csv = _resolve_path(source or Path(experiment_config.prompts_source_csv))
    output_jsonl = _resolve_path(output or Path(experiment_config.prompts_file))
    overrides_path = _resolve_path(overrides or Path(experiment_config.prompts_overrides))

    if not source_csv.exists():
        console.print(f"[red]Source CSV not found:[/red] {source_csv}")
        raise typer.Exit(code=1)

    prompts = import_prompts_from_csv(
        source_csv=source_csv,
        output_jsonl=output_jsonl,
        overrides_path=overrides_path if overrides_path.exists() else None,
        defaults=experiment_config.prompt_defaults,
    )

    language_counts: dict[str, int] = {}
    for prompt in prompts:
        language_counts[prompt.language.value] = language_counts.get(prompt.language.value, 0) + 1

    console.print("[bold green]Prompt import complete[/bold green]")
    console.print(f"Source CSV: {source_csv}")
    console.print(f"Output JSONL: {output_jsonl}")
    console.print(f"Total prompts: {len(prompts)}")
    console.print(f"Unique base prompts: {len({prompt.base_prompt_id for prompt in prompts})}")
    if overrides_path.exists():
        console.print(f"Overrides applied from: {overrides_path}")
    console.print(f"By language: {language_counts}")


if __name__ == "__main__":
    app()
