import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from src.config.loader import load_experiment_config
from src.data.import_prompts_txt import import_prompts_txt

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
    source: Path | None = typer.Option(None, "--source", help="Plain-text prompts file, one prompt per line."),
    output: Path | None = typer.Option(None, "--output", help="Processed prompts JSONL output."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Experiment config YAML."),
) -> None:
    experiment_config = load_experiment_config(_resolve_path(config))
    source_txt = _resolve_path(source or Path(experiment_config.prompts_source_txt))
    output_jsonl = _resolve_path(output or Path(experiment_config.prompts_file))

    if not source_txt.exists():
        console.print(f"[red]Source TXT not found:[/red] {source_txt}")
        raise typer.Exit(code=1)

    prompts = import_prompts_txt(source_txt, output_jsonl=output_jsonl)

    language_counts: dict[str, int] = {}
    for prompt in prompts:
        language_counts[prompt.language.value] = language_counts.get(prompt.language.value, 0) + 1

    console.print("[bold green]TXT import complete[/bold green]")
    console.print(f"Source TXT: {source_txt}")
    console.print(f"Output JSONL: {output_jsonl}")
    console.print(f"Total prompts: {len(prompts)}")
    console.print(f"Unique base prompts: {len({prompt.base_prompt_id for prompt in prompts})}")
    console.print(f"By language: {language_counts}")


if __name__ == "__main__":
    app()
