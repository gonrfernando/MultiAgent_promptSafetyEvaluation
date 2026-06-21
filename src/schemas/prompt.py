from enum import Enum

from pydantic import BaseModel, Field


class LanguageVariant(str, Enum):
    EN = "en"
    ES_NEUTRAL = "es_neutral"
    ES_MX_GDL = "es_mx_gdl"
    ES_AR_CABA = "es_ar_caba"
    ES_CL_SANTIAGO = "es_cl_santiago"
    ES_ES_BCN = "es_es_bcn"
    ES_UY = "es_uy"


class PromptCategory(str, Enum):
    BENIGN = "benign"
    AMBIGUOUS = "ambiguous"
    HARMFUL = "harmful"
    JAILBREAK = "jailbreak"


class AdaptationMethod(str, Enum):
    ORIGINAL = "original"
    AI_NEUTRAL = "ai_neutral"
    AI_REGIONAL = "ai_regional"
    MANUAL = "manual"


class Prompt(BaseModel):
    prompt_id: str
    base_prompt_id: str
    language: LanguageVariant
    text: str
    category: PromptCategory | None = None
    expected_risk: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional prior risk estimate; omitted for judge runs to avoid bias",
    )
    source: str = "harmbench"
    source_index: int = Field(default=1, ge=1, description="Row number in the source file")
    harmbench_behavior: str | None = None
    adaptation_method: AdaptationMethod = AdaptationMethod.ORIGINAL
    region: str | None = None
    semantic_equivalence_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence that language variants preserve intent",
    )
    is_manual_override: bool = False
