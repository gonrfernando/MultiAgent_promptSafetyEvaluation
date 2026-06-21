"""Generate charts and summary tables from a judge evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "evaluations" / "judges-full"

LANGUAGE_ORDER = [
    "es_neutral",
    "es_mx_gdl",
    "es_ar_caba",
    "es_cl_santiago",
    "es_es_bcn",
    "es_uy",
]

LANGUAGE_LABELS = {
    "es_neutral": "Neutral",
    "es_mx_gdl": "GDL (MX)",
    "es_ar_caba": "CABA (AR)",
    "es_cl_santiago": "Santiago (CL)",
    "es_es_bcn": "Barcelona (ES)",
    "es_uy": "Uruguay (UY)",
}


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    evaluations_path = run_dir / "evaluations.jsonl"
    metrics_path = run_dir / "disagreement_metrics.jsonl"
    metadata_path = run_dir / "run_metadata.json"

    if not evaluations_path.exists():
        raise FileNotFoundError(f"Missing evaluations file: {evaluations_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing disagreement metrics file: {metrics_path}")

    evaluations = pd.read_json(evaluations_path, lines=True)
    metrics = pd.read_json(metrics_path, lines=True)

    prompt_meta = evaluations[["prompt_id", "base_prompt_id", "language"]].drop_duplicates()
    metrics = metrics.merge(prompt_meta, left_on="subject_id", right_on="prompt_id", how="left")
    metrics = metrics.drop(columns=["prompt_id"])

    metadata: dict = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return evaluations, metrics, metadata


def _ordered_languages(values: pd.Series) -> list[str]:
    present = [language for language in LANGUAGE_ORDER if language in set(values)]
    extras = sorted(set(values) - set(present))
    return present + extras


def plot_entropy_by_language(metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        metrics.groupby("language", observed=True)["vote_entropy"]
        .mean()
        .reindex(_ordered_languages(metrics["language"]))
    )
    labels = [LANGUAGE_LABELS.get(language, language) for language in grouped.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Mean judge disagreement by language")
    ax.set_xlabel("Language variant")
    ax.set_ylabel("Vote entropy")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, max(1.0, grouped.max() * 1.15))
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_by_language.png", dpi=150)
    plt.close(fig)


def plot_risk_by_language(metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        metrics.groupby("language", observed=True)["mean_risk_score"]
        .mean()
        .reindex(_ordered_languages(metrics["language"]))
    )
    labels = [LANGUAGE_LABELS.get(language, language) for language in grouped.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_title("Mean risk score by language")
    ax.set_xlabel("Language variant")
    ax.set_ylabel("Mean risk score")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_by_language.png", dpi=150)
    plt.close(fig)


def plot_variance_by_language(metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        metrics.groupby("language", observed=True)["risk_score_variance"]
        .mean()
        .reindex(_ordered_languages(metrics["language"]))
    )
    labels = [LANGUAGE_LABELS.get(language, language) for language in grouped.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Mean risk score variance by language")
    ax.set_xlabel("Language variant")
    ax.set_ylabel("Risk score variance")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "variance_by_language.png", dpi=150)
    plt.close(fig)


def plot_heatmap_entropy(metrics: pd.DataFrame, output_dir: Path) -> None:
    pivot = metrics.pivot(index="base_prompt_id", columns="language", values="vote_entropy")
    pivot = pivot.reindex(columns=_ordered_languages(metrics["language"]))
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 12))
    image = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_title("Disagreement heatmap (vote entropy)")
    ax.set_xlabel("Language variant")
    ax.set_ylabel("Base prompt")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([LANGUAGE_LABELS.get(column, column) for column in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    fig.colorbar(image, ax=ax, label="Vote entropy")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_entropy.png", dpi=150)
    plt.close(fig)


def plot_votes_by_judge(evaluations: pd.DataFrame, output_dir: Path) -> None:
    counts = evaluations.groupby(["judge_id", "safe"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(index=["security", "policy", "critical"])

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    ax.set_title("Safe vs unsafe votes by judge")
    ax.set_xlabel("Judge")
    ax.set_ylabel("Count")
    ax.legend(title="safe", labels=["False (unsafe)", "True (safe)"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "votes_by_judge.png", dpi=150)
    plt.close(fig)


def plot_entropy_vs_variance(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for language in _ordered_languages(metrics["language"]):
        subset = metrics[metrics["language"] == language]
        ax.scatter(
            subset["vote_entropy"],
            subset["risk_score_variance"],
            label=LANGUAGE_LABELS.get(language, language),
            alpha=0.75,
            s=35,
        )
    ax.set_title("Disagreement entropy vs risk score variance")
    ax.set_xlabel("Vote entropy")
    ax.set_ylabel("Risk score variance")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_vs_variance_scatter.png", dpi=150)
    plt.close(fig)


def write_summary_tables(metrics: pd.DataFrame, evaluations: pd.DataFrame, output_dir: Path) -> None:
    by_language = metrics.groupby("language", observed=True).agg(
        prompt_count=("subject_id", "count"),
        mean_vote_entropy=("vote_entropy", "mean"),
        mean_pairwise_disagreement=("pairwise_disagreement_rate", "mean"),
        mean_risk_score=("mean_risk_score", "mean"),
        mean_risk_variance=("risk_score_variance", "mean"),
        disagreement_prompts=("vote_entropy", lambda values: int((values > 0).sum())),
    )
    by_language = by_language.reindex(_ordered_languages(metrics["language"]))
    by_language.index = [LANGUAGE_LABELS.get(index, index) for index in by_language.index]
    by_language.to_csv(output_dir / "summary_by_language.csv")

    top_disagreement = metrics.sort_values(
        ["vote_entropy", "pairwise_disagreement_rate", "risk_score_variance"],
        ascending=False,
    ).head(15)
    top_disagreement.to_csv(output_dir / "top_disagreement_prompts.csv", index=False)

    judge_summary = evaluations.groupby(["judge_id", "safe"], observed=True).size().unstack(fill_value=0)
    judge_summary.to_csv(output_dir / "summary_by_judge.csv")


def print_console_summary(metrics: pd.DataFrame, metadata: dict) -> None:
    print("=" * 60)
    print("Judge evaluation analysis")
    if metadata:
        print(f"Run ID: {metadata.get('run_id', 'unknown')}")
        print(f"Prompts: {metadata.get('prompt_count', len(metrics))}")
        print(f"Evaluations: {metadata.get('completed_evaluations', 'unknown')}")
        print(f"Refusals / parse failures: {metadata.get('refused_or_parse_failures', 'unknown')}")
    print(f"Prompts with any disagreement: {(metrics['vote_entropy'] > 0).sum()} / {len(metrics)}")
    print("=" * 60)
    print("\nTop 10 disagreement prompts:")
    columns = [
        "subject_id",
        "language",
        "vote_entropy",
        "pairwise_disagreement_rate",
        "mean_risk_score",
        "majority_vote_safe",
    ]
    print(
        metrics.sort_values("vote_entropy", ascending=False)
        .head(10)[columns]
        .to_string(index=False)
    )
    print("\nMean vote entropy by language:")
    print(
        metrics.groupby("language", observed=True)["vote_entropy"]
        .mean()
        .reindex(_ordered_languages(metrics["language"]))
        .rename(index=LANGUAGE_LABELS)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judge evaluation run and generate charts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Path to evaluation run directory (default: judges-full)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures and CSV summaries (default: <run-dir>/figures)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = args.output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations, metrics, metadata = load_run(run_dir)

    plot_entropy_by_language(metrics, output_dir)
    plot_risk_by_language(metrics, output_dir)
    plot_variance_by_language(metrics, output_dir)
    plot_heatmap_entropy(metrics, output_dir)
    plot_votes_by_judge(evaluations, output_dir)
    plot_entropy_vs_variance(metrics, output_dir)
    write_summary_tables(metrics, evaluations, output_dir)

    print_console_summary(metrics, metadata)
    print(f"\nFigures saved to: {output_dir}")
    print(f"CSV summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
