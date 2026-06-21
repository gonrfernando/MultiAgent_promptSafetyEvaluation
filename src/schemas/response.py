from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.schemas.prompt import LanguageVariant


class GeneratedResponse(BaseModel):
    response_id: str
    prompt_id: str
    base_prompt_id: str
    language: LanguageVariant
    model: str
    text: str
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generation_temperature: float = Field(ge=0.0, le=2.0)
    generation_max_tokens: int = Field(ge=1)
