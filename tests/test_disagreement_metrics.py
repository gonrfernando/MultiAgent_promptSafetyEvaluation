import pytest

from src.disagreement.metrics import (
    compute_disagreement_metrics,
    majority_vote_safe,
    pairwise_disagreement_rate,
    risk_score_range,
    risk_score_variance,
    vote_entropy,
)
from src.schemas.evaluation import AgentEvaluation, EvaluatorRole


def _evaluation(role: EvaluatorRole, safe: bool, risk_score: float) -> AgentEvaluation:
    return AgentEvaluation(
        evaluation_id=f"eval-{role.value}",
        response_id="resp-001",
        agent_id=role.value,
        role=role,
        safe=safe,
        risk_score=risk_score,
        explanation="test",
        category="test",
        model="mock-model",
    )


def test_vote_entropy_unanimous_safe():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.1),
        _evaluation(EvaluatorRole.POLICY, True, 0.1),
        _evaluation(EvaluatorRole.CRITICAL, True, 0.1),
    ]

    assert vote_entropy(evaluations) == pytest.approx(0.0)


def test_vote_entropy_max_disagreement():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.1),
        _evaluation(EvaluatorRole.POLICY, False, 0.9),
        _evaluation(EvaluatorRole.CRITICAL, False, 0.8),
    ]

    assert vote_entropy(evaluations) == pytest.approx(1.0)


def test_pairwise_disagreement_rate():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.2),
        _evaluation(EvaluatorRole.POLICY, False, 0.8),
        _evaluation(EvaluatorRole.CRITICAL, False, 0.7),
    ]

    assert pairwise_disagreement_rate(evaluations) == pytest.approx(4 / 6)


def test_risk_score_variance_and_range():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.3),
        _evaluation(EvaluatorRole.POLICY, False, 0.5),
        _evaluation(EvaluatorRole.CRITICAL, False, 0.9),
    ]

    assert risk_score_range(evaluations) == pytest.approx(0.8)
    assert risk_score_variance(evaluations) == pytest.approx(0.0875)


def test_majority_vote_safe():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.2),
        _evaluation(EvaluatorRole.POLICY, False, 0.8),
        _evaluation(EvaluatorRole.CRITICAL, False, 0.7),
    ]

    assert majority_vote_safe(evaluations) is True


def test_compute_disagreement_metrics():
    evaluations = [
        _evaluation(EvaluatorRole.SECURITY, True, 0.1),
        _evaluation(EvaluatorRole.ETHICS, True, 0.2),
        _evaluation(EvaluatorRole.POLICY, False, 0.8),
        _evaluation(EvaluatorRole.CRITICAL, False, 0.7),
    ]

    metrics = compute_disagreement_metrics("resp-001", evaluations)

    assert metrics.subject_id == "resp-001"
    assert metrics.evaluator_count == 4
    assert metrics.majority_vote_safe is True
    assert metrics.mean_risk_score == pytest.approx(0.45)
    assert metrics.vote_entropy == pytest.approx(1.0)


def test_compute_disagreement_metrics_requires_evaluations():
    with pytest.raises(ValueError):
        compute_disagreement_metrics("resp-001", [])
