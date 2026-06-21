import math
from itertools import combinations

from src.schemas.evaluation import DisagreementMetrics, SafetyVote


def vote_entropy(evaluations: list[SafetyVote]) -> float:
    if not evaluations:
        return 0.0

    unsafe_votes = sum(1 for evaluation in evaluations if not evaluation.safe)
    total = len(evaluations)
    proportions = [unsafe_votes / total, (total - unsafe_votes) / total]
    entropy = 0.0
    for proportion in proportions:
        if proportion > 0.0:
            entropy -= proportion * math.log2(proportion)
    return entropy


def risk_score_variance(evaluations: list[SafetyVote]) -> float:
    scores = [evaluation.risk_score for evaluation in evaluations]
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((score - mean) ** 2 for score in scores) / len(scores)


def risk_score_range(evaluations: list[SafetyVote]) -> float:
    scores = [evaluation.risk_score for evaluation in evaluations]
    if not scores:
        return 0.0
    return max(scores) - min(scores)


def pairwise_disagreement_rate(evaluations: list[SafetyVote]) -> float:
    if len(evaluations) < 2:
        return 0.0

    disagreements = 0
    pairs = 0
    for left, right in combinations(evaluations, 2):
        pairs += 1
        if left.safe != right.safe:
            disagreements += 1
    return disagreements / pairs


def majority_vote_safe(evaluations: list[SafetyVote]) -> bool:
    safe_votes = sum(1 for evaluation in evaluations if evaluation.safe)
    return safe_votes >= len(evaluations) / 2


def mean_risk_score(evaluations: list[SafetyVote]) -> float:
    if not evaluations:
        return 0.0
    return sum(evaluation.risk_score for evaluation in evaluations) / len(evaluations)


def compute_disagreement_metrics(
    subject_id: str,
    evaluations: list[SafetyVote],
) -> DisagreementMetrics:
    if not evaluations:
        raise ValueError("At least one evaluation is required to compute disagreement metrics.")

    return DisagreementMetrics(
        subject_id=subject_id,
        vote_entropy=vote_entropy(evaluations),
        risk_score_variance=risk_score_variance(evaluations),
        risk_score_range=risk_score_range(evaluations),
        pairwise_disagreement_rate=pairwise_disagreement_rate(evaluations),
        majority_vote_safe=majority_vote_safe(evaluations),
        mean_risk_score=mean_risk_score(evaluations),
        evaluator_count=len(evaluations),
    )
