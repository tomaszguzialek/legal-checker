"""Deterministic post-validation of the LLM assessment (SPEC Task 4.4)."""

import re
from datetime import date

from langchain_core.messages import AnyMessage, ToolMessage

from legal_checker.models import ActDocument, ComplianceResult, LegalBasis, LegalUnit, LlmEvaluationPayload

PARAPHRASED_EXCERPT_WARNING = (
    "Cytat z dokumentu nie występuje dosłownie we fragmencie wejściowym (możliwa parafraza)"
)
# A quote the model did not copy verbatim makes the whole assessment less trustworthy.
PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR = 0.5
UNCLEAR_APPLICABILITY_WARNING = "Nie wiadomo, czy akt ma zastosowanie do stron — pewność ograniczona do {cap:.1f}"
# The highest confidence of an assessment that may rest on an act which does not apply to the parties.
UNCLEAR_APPLICABILITY_MAX_CONFIDENCE = 0.7
UNCONSULTED_PROVISION_WARNING = "Wskazany przepis „{provision}” nie należy do przepisów widzianych przez agenta"

# Unit numbers as the chunker writes them: "27", "43a", "449^1".
_NUMBER = r"\d+[a-z]*(?:\^\d+[a-z]*)?"
_ARTICLE_REFERENCE = re.compile(rf"\bart\.\s*({_NUMBER})", re.IGNORECASE)
_PARAGRAPH_REFERENCE = re.compile(rf"§\s*({_NUMBER})")


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


def provision_unit(provision: str) -> tuple[str, str] | None:
    """The unit a provision cites, as ``(prefix, number)`` of its ``unit_id``, or ``None`` if it cites none.

    "art. 12 ust. 1 pkt 2" -> ``("art", "12")``; "art. 8 § 1" (a code) -> ``("art", "8")``;
    "§ 5 ust. 2" (a regulation without articles) -> ``("par", "5")``.
    """
    if match := _ARTICLE_REFERENCE.search(provision):
        return "art", match[1].lower()
    if match := _PARAGRAPH_REFERENCE.search(provision):
        return "par", match[1].lower()
    return None


def unit_of_id(unit_id: str) -> tuple[str, str]:
    """``(prefix, number)`` of the article or "§" unit a ``unit_id`` belongs to.

    "DU/2014/827:art.27.ust.2" -> ``("art", "27")``.
    """
    prefix, number, *_ = unit_id.rsplit(":", 1)[-1].split(".")
    return prefix, number.lower()


def provision_was_consulted(provision: str, unit_ids: list[str]) -> bool:
    """Whether the article (or "§" unit) a provision cites is among the units the agent has seen."""
    cited = provision_unit(provision)
    return cited is not None and cited in {unit_of_id(unit_id) for unit_id in unit_ids}


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

    The act's title, ELI and source URL, the consulted units and the date of the assessment
    (``as_of_date``, today) come from code, never from the LLM.

    When the LLM flags that it is unclear whether the act applies to the parties, the confidence is capped
    at ``UNCLEAR_APPLICABILITY_MAX_CONFIDENCE``; if that lowers it, a warning is added.
    A ``document_excerpt`` not found verbatim in ``document_fragment`` adds a warning and lowers the confidence.
    A ``matched_provision`` whose article is not among the consulted units adds a warning.
    """
    confidence = payload.confidence
    warnings = list(warnings)
    if payload.applicability_unclear and confidence > UNCLEAR_APPLICABILITY_MAX_CONFIDENCE:
        warnings.append(UNCLEAR_APPLICABILITY_WARNING.format(cap=UNCLEAR_APPLICABILITY_MAX_CONFIDENCE))
        confidence = UNCLEAR_APPLICABILITY_MAX_CONFIDENCE
    if payload.document_excerpt is not None and not excerpt_is_verbatim(payload.document_excerpt, document_fragment):
        warnings.append(PARAPHRASED_EXCERPT_WARNING)
        confidence *= PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR
    unit_ids = consulted_units(candidates, messages)
    provision = payload.matched_provision
    if provision is not None and not provision_was_consulted(provision, unit_ids):
        warnings.append(UNCONSULTED_PROVISION_WARNING.format(provision=provision))
    return ComplianceResult(
        status=payload.status,
        confidence=confidence,
        document_excerpt=payload.document_excerpt,
        legal_basis=LegalBasis(
            act_title=act.title,
            eli=act.eli,
            provision=payload.matched_provision,
            as_of_date=date.today().isoformat(),
            source_url=act.source_url,
        ),
        rationale=payload.rationale,
        suggested_fix=payload.suggested_fix,
        consulted_units=unit_ids,
        warnings=warnings,
    )
