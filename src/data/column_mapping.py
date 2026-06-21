from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.prompt import AdaptationMethod, LanguageVariant


def normalize_header(header: str) -> str:
    return re.sub(r"\s+", " ", header.strip().lower())


@dataclass(frozen=True)
class ColumnSpec:
    language: LanguageVariant
    adaptation_method: AdaptationMethod
    region: str | None = None


COLUMN_SPECS: dict[str, ColumnSpec] = {
    "prompt original (en)": ColumnSpec(LanguageVariant.EN, AdaptationMethod.ORIGINAL),
    "prompt original en": ColumnSpec(LanguageVariant.EN, AdaptationMethod.ORIGINAL),
    "english": ColumnSpec(LanguageVariant.EN, AdaptationMethod.ORIGINAL),
    "en": ColumnSpec(LanguageVariant.EN, AdaptationMethod.ORIGINAL),
    "neutral spanish": ColumnSpec(LanguageVariant.ES_NEUTRAL, AdaptationMethod.AI_NEUTRAL),
    "es neutral": ColumnSpec(LanguageVariant.ES_NEUTRAL, AdaptationMethod.AI_NEUTRAL),
    "spanish neutral": ColumnSpec(LanguageVariant.ES_NEUTRAL, AdaptationMethod.AI_NEUTRAL),
    "mexican spanish (gdl, jal.)": ColumnSpec(
        LanguageVariant.ES_MX_GDL,
        AdaptationMethod.AI_REGIONAL,
        "gdl",
    ),
    "mexican spanish": ColumnSpec(
        LanguageVariant.ES_MX_GDL,
        AdaptationMethod.AI_REGIONAL,
        "gdl",
    ),
    "es mx gdl": ColumnSpec(LanguageVariant.ES_MX_GDL, AdaptationMethod.AI_REGIONAL, "gdl"),
    "argentinian spanish (caba)": ColumnSpec(
        LanguageVariant.ES_AR_CABA,
        AdaptationMethod.AI_REGIONAL,
        "caba",
    ),
    "argentinian spanish": ColumnSpec(
        LanguageVariant.ES_AR_CABA,
        AdaptationMethod.AI_REGIONAL,
        "caba",
    ),
    "es ar caba": ColumnSpec(LanguageVariant.ES_AR_CABA, AdaptationMethod.AI_REGIONAL, "caba"),
    "chilean spanish (santiago)": ColumnSpec(
        LanguageVariant.ES_CL_SANTIAGO,
        AdaptationMethod.AI_REGIONAL,
        "santiago",
    ),
    "chilean spanish": ColumnSpec(
        LanguageVariant.ES_CL_SANTIAGO,
        AdaptationMethod.AI_REGIONAL,
        "santiago",
    ),
    "es cl santiago": ColumnSpec(
        LanguageVariant.ES_CL_SANTIAGO,
        AdaptationMethod.AI_REGIONAL,
        "santiago",
    ),
    "spanish from spain (bcn)": ColumnSpec(
        LanguageVariant.ES_ES_BCN,
        AdaptationMethod.AI_REGIONAL,
        "bcn",
    ),
    "spanish from spain": ColumnSpec(
        LanguageVariant.ES_ES_BCN,
        AdaptationMethod.AI_REGIONAL,
        "bcn",
    ),
    "es es bcn": ColumnSpec(LanguageVariant.ES_ES_BCN, AdaptationMethod.AI_REGIONAL, "bcn"),
}

OPTIONAL_METADATA_COLUMNS = {
    "source_index",
    "harmbench_behavior",
    "category",
    "expected_risk",
    "semantic_equivalence_confidence",
}


def resolve_column_spec(header: str) -> ColumnSpec | None:
    normalized = normalize_header(header)
    if normalized in OPTIONAL_METADATA_COLUMNS:
        return None
    if normalized in COLUMN_SPECS:
        return COLUMN_SPECS[normalized]

    for key, spec in COLUMN_SPECS.items():
        if key in normalized or normalized in key:
            return spec
    return None
