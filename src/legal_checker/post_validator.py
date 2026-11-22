"""Deterministic post-validation of the LLM assessment (SPEC Task 4.4)."""

import re

from langchain_core.messages import AnyMessage, ToolMessage

from legal_checker.models import ActDocument, ComplianceResult, LegalBasis, LegalUnit, LlmEvaluationPayload

PARAPHRASED_EXCERPT_WARNING = "Cytat z dokumentu nie występuje dosłownie we fragmencie wejściowym (możliwa parafraza)"
# A quote the model did not copy verbatim makes the whole assessment less trustworthy.
PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR = 0.5


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt_is_verbatim(excerpt: str, document_fragment: str) -> bool:
    """Whether the excerpt occurs in the fragment, ignoring differences in whitespace."""
    return normalize_whitespace(excerpt) in normalize_whitespace(document_fragment)


def consulted_units(candidates: list[tuple[LegalUnit, float]], messages: list[AnyMessage]) -> list[str]:
    """``unit_id``s the agent has seen: BM25 candidates, then tool results, in order and without repeats.

    Derived from the tools' artifacts rather than from anything the LLM claims to have read.
    """
    unit_ids = [unit.unit_id for unit, _ in candidates]
    for message in messages:
        if isinstance(message, ToolMessage):
            unit_ids.extend(message.artifact or [])
    return list(dict.fromkeys(unit_ids))


def post_validate(
    payload: LlmEvaluationPayload,
    *,
    document_fragment: str,
    act: ActDocument,
    candidates: list[tuple[LegalUnit, float]],
    messages: list[AnyMessage],
    warnings: list[str],
) -> ComplianceResult:
    """The final result: the LLM's assessment with source metadata attached by code.

    A ``document_excerpt`` not found verbatim in ``document_fragment`` adds a warning and lowers the confidence.
    """
    confidence = payload.confidence
    warnings = list(warnings)
    if payload.document_excerpt is not None and not excerpt_is_verbatim(payload.document_excerpt, document_fragment):
        warnings.append(PARAPHRASED_EXCERPT_WARNING)
        confidence *= PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR
    return ComplianceResult(
        status=payload.status,
        confidence=confidence,
        document_excerpt=payload.document_excerpt,
        legal_basis=LegalBasis(
            act_title=act.title,
            eli=act.eli,
            provision=payload.matched_provision,
            as_of_date=act.as_of_date,
            source_url=act.source_url,
        ),
        rationale=payload.rationale,
        suggested_fix=payload.suggested_fix,
        consulted_units=consulted_units(candidates, messages),
        warnings=warnings,
    )
