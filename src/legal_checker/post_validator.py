"""Deterministic post-validation of the LLM assessment (SPEC Task 4.4)."""

from langchain_core.messages import AnyMessage, ToolMessage

from legal_checker.models import ActDocument, ComplianceResult, LegalBasis, LegalUnit, LlmEvaluationPayload


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
    act: ActDocument,
    candidates: list[tuple[LegalUnit, float]],
    messages: list[AnyMessage],
    warnings: list[str],
) -> ComplianceResult:
    """The final result: the LLM's assessment with source metadata attached by code."""
    return ComplianceResult(
        status=payload.status,
        confidence=payload.confidence,
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
        warnings=list(warnings),
    )
