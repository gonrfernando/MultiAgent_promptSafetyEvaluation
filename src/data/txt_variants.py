from dataclasses import dataclass

from src.schemas.prompt import AdaptationMethod, LanguageVariant


@dataclass(frozen=True)
class TxtVariantSpec:
    language: LanguageVariant
    adaptation_method: AdaptationMethod
    region: str | None


TXT_VARIANT_ORDER: list[TxtVariantSpec] = [
    TxtVariantSpec(LanguageVariant.ES_NEUTRAL, AdaptationMethod.AI_NEUTRAL, None),
    TxtVariantSpec(LanguageVariant.ES_MX_GDL, AdaptationMethod.AI_REGIONAL, "gdl"),
    TxtVariantSpec(LanguageVariant.ES_AR_CABA, AdaptationMethod.AI_REGIONAL, "caba"),
    TxtVariantSpec(LanguageVariant.ES_CL_SANTIAGO, AdaptationMethod.AI_REGIONAL, "santiago"),
    TxtVariantSpec(LanguageVariant.ES_ES_BCN, AdaptationMethod.AI_REGIONAL, "bcn"),
    TxtVariantSpec(LanguageVariant.ES_UY, AdaptationMethod.AI_REGIONAL, "uy"),
]

VARIANTS_PER_BLOCK = len(TXT_VARIANT_ORDER)
