import json
from pathlib import Path

from typer.testing import CliRunner

from src.pipeline.run_experiment import app

runner = CliRunner()


def test_dry_run_writes_jsonl_outputs(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    config_text = config_path.read_text(encoding="utf-8").replace(
        "results_dir: results/runs",
        f"results_dir: {tmp_path.as_posix()}",
    )
    temp_config = tmp_path / "experiment.yaml"
    temp_config.write_text(config_text, encoding="utf-8")

    result = runner.invoke(app, ["--dry-run", "--config", str(temp_config), "--run-id", "test-run"])

    assert result.exit_code == 0

    run_dir = tmp_path / "test-run"
    assert (run_dir / "prompts.jsonl").exists()
    assert (run_dir / "responses.jsonl").exists()
    assert (run_dir / "evaluations.jsonl").exists()
    assert (run_dir / "disagreement_metrics.jsonl").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "metrics.json").exists()

    with (run_dir / "metrics.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["prompt_count"] == 3
    assert summary["response_count"] == 3
    assert summary["evaluation_count"] == 12
    assert summary["disagreement_records"] == 3
