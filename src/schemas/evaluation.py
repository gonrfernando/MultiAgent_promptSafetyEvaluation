from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from src.schemas.prompt import LanguageVariant


class EvaluatorRole(str, Enum):
    SECURITY = "security"
    ETHICS = "ethics"
    POLICY = "policy"
    CRITICAL = "critical"


class JudgeRole(str, Enum):
    SECURITY = "security"
    POLICY = "policy"
    CRITICAL = "critical"


class SafetyVote(Protocol):
    safe: bool
    risk_score: float


class AgentEvaluation(BaseModel):
    evaluation_id: str
    response_id: str
    agent_id: str
    role: EvaluatorRole
    safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    category: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptEvaluation(BaseModel):
    evaluation_id: str
    prompt_id: str
    base_prompt_id: str
    language: LanguageVariant
    judge_id: str
    role: JudgeRole
    safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    risk_category: str = ""
    model: str
    judge_refused: bool = False
    parse_error: str | None = None
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DisagreementMetrics(BaseModel):
    subject_id: str
    vote_entropy: float = Field(ge=0.0)
    risk_score_variance: float = Field(ge=0.0)
    risk_score_range: float = Field(ge=0.0)
    pairwise_disagreement_rate: float = Field(ge=0.0, le=1.0)
    majority_vote_safe: bool
    mean_risk_score: float = Field(ge=0.0, le=1.0)
    evaluator_count: int = Field(ge=1)

    @property
    def response_id(self) -> str:
        return self.subject_id
