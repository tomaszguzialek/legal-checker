from collections.abc import Sequence
from datetime import date

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from legal_checker.models import ActDocument, ComplianceStatus, LegalUnit, LlmEvaluationPayload
from legal_checker.post_validator import (
    PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR,
    PARAPHRASED_EXCERPT_WARNING,
    UNCLEAR_APPLICABILITY_MAX_CONFIDENCE,
    UNCLEAR_APPLICABILITY_WARNING,
    UNCONSULTED_PROVISION_WARNING,
    consulted_units,
    excerpt_is_verbatim,
    post_validate,
    provision_unit,
    unit_of_id,
)

FRAGMENT = (
    "§ 7. Klient może odstąpić od umowy zawartej na odległość\n"
    "w terminie 14 dni   bez podania przyczyny, składając oświadczenie na formularzu."
)
ACT = ActDocument(
    eli="DU/2014/827",
    title="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
    source_url="https://eli-api.test/eli/acts/DU/2014/827/text/U/D20140827Lj.pdf",
    as_of_date="2026-01-01",
    text="",
)


def unit(unit_id: str) -> LegalUnit:
    article = unit_of_id(unit_id)[1]
    return LegalUnit(unit_id=unit_id, eli="DU/2014/827", article=article, hierarchy_path=[], text=f"Art. {article}.")


CANDIDATES = [(unit("DU/2014/827:art.27"), 4.2), (unit("DU/2014/827:art.38.ust.1"), 1.3)]


def payload(**fields) -> LlmEvaluationPayload:
    defaults = {
        "status": ComplianceStatus.COMPLIANT,
        "applicability_unclear": False,
        "confidence": 0.8,
        "document_excerpt": "w terminie 14 dni bez podania przyczyny",
        "matched_provision": "art. 27",
        "rationale": "Zapis powtarza ustawowy termin odstąpienia.",
        "suggested_fix": None,
    }
    return LlmEvaluationPayload(**{**defaults, **fields})


def run(llm_payload: LlmEvaluationPayload, messages: Sequence[AnyMessage] = (), warnings: Sequence[str] = ()):
    return post_validate(
        llm_payload,
        document_fragment=FRAGMENT,
        act=ACT,
        candidates=CANDIDATES,
        messages=list(messages),
        warnings=list(warnings),
    )


def tool_result(call_id: str, unit_ids: list[str]) -> ToolMessage:
    return ToolMessage("[…]", tool_call_id=call_id, artifact=unit_ids)


# Excerpt check (4.4.1)


@pytest.mark.parametrize(
    "excerpt",
    [
        "w terminie 14 dni bez podania przyczyny",
        "na odległość w terminie 14 dni",  # across a line break
        "  w terminie 14 dni\tbez podania przyczyny ",
        FRAGMENT,
    ],
)
def test_excerpt_in_the_fragment_is_verbatim_up_to_whitespace(excerpt):
    assert excerpt_is_verbatim(excerpt, FRAGMENT)


@pytest.mark.parametrize(
    "excerpt",
    [
        "w ciągu 14 dni bez podawania przyczyny",  # paraphrase
        "w terminie 7 dni",  # invented
        "W terminie 14 dni",  # different case
        "w terminie 14dni",  # whitespace removed, not just normalised
    ],
)
def test_excerpt_not_in_the_fragment_is_not_verbatim(excerpt):
    assert not excerpt_is_verbatim(excerpt, FRAGMENT)


def test_verbatim_excerpt_keeps_confidence_and_adds_no_warning():
    result = run(payload())

    assert result.confidence == 0.8
    assert result.warnings == []


def test_invented_excerpt_adds_a_warning_and_lowers_confidence():
    result = run(payload(document_excerpt="Klient ma 7 dni na odstąpienie od umowy."))

    assert result.warnings == [PARAPHRASED_EXCERPT_WARNING]
    assert result.confidence == pytest.approx(0.8 * PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR)
    assert result.document_excerpt == "Klient ma 7 dni na odstąpienie od umowy."  # kept for the reader to compare


def test_missing_excerpt_is_not_checked():
    result = run(payload(document_excerpt=None))

    assert result.confidence == 0.8
    assert result.warnings == []


# Unclear applicability to the parties

CAPPED_WARNING = UNCLEAR_APPLICABILITY_WARNING.format(cap=UNCLEAR_APPLICABILITY_MAX_CONFIDENCE)


@pytest.mark.parametrize("status", list(ComplianceStatus))
def test_unclear_applicability_caps_the_confidence_of_every_status(status):
    result = run(payload(status=status, applicability_unclear=True, confidence=0.95))

    assert result.confidence == UNCLEAR_APPLICABILITY_MAX_CONFIDENCE
    assert result.warnings == [CAPPED_WARNING]


@pytest.mark.parametrize("confidence", [0.4, UNCLEAR_APPLICABILITY_MAX_CONFIDENCE])
def test_unclear_applicability_keeps_a_confidence_within_the_cap_without_a_warning(confidence):
    result = run(payload(applicability_unclear=True, confidence=confidence))

    assert result.confidence == confidence
    assert result.warnings == []


def test_clear_applicability_keeps_a_high_confidence():
    assert run(payload(confidence=0.95)).confidence == 0.95


def test_paraphrased_excerpt_lowers_the_capped_confidence_further():
    result = run(payload(applicability_unclear=True, confidence=0.95, document_excerpt="termin 7 dni"))

    assert result.confidence == pytest.approx(UNCLEAR_APPLICABILITY_MAX_CONFIDENCE * PARAPHRASED_EXCERPT_CONFIDENCE_FACTOR)
    assert result.warnings == [CAPPED_WARNING, PARAPHRASED_EXCERPT_WARNING]


# Consulted units and the provision check (4.4.2)


def test_consulted_units_are_candidates_then_tool_artifacts_without_repeats():
    messages = [
        AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "30"}, "id": "c1"}]),
        tool_result("c1", ["DU/2014/827:art.30", "DU/2014/827:art.27"]),
        AIMessage("", tool_calls=[{"name": "search_provisions", "args": {"query": "zwrot"}, "id": "c2"}]),
        tool_result("c2", []),  # nothing found
        ToolMessage("Nie wykonano", tool_call_id="c3", artifact=None),  # no artifact at all
        AIMessage("art.99 według mnie też ma zastosowanie"),  # the model's claims don't count
    ]

    assert consulted_units(CANDIDATES, messages) == [
        "DU/2014/827:art.27",
        "DU/2014/827:art.38.ust.1",
        "DU/2014/827:art.30",
    ]


@pytest.mark.parametrize(
    ("provision", "expected"),
    [
        ("art. 27", ("art", "27")),
        ("Art. 12 ust. 1 pkt 2", ("art", "12")),
        ("art.43a ust. 2", ("art", "43a")),
        ("art. 449^1", ("art", "449^1")),
        ("art. 8 § 1", ("art", "8")),  # a paragraph of a code's article
        ("§ 5 ust. 2", ("par", "5")),  # a unit of a regulation without articles
        ("przepisy ogólne", None),
    ],
)
def test_provision_unit(provision, expected):
    assert provision_unit(provision) == expected


@pytest.mark.parametrize(
    ("unit_id", "expected"),
    [
        ("DU/2014/827:art.27", ("art", "27")),
        ("DU/2014/827:art.27.ust.2", ("art", "27")),
        ("DU/1964/16:art.8.par.1", ("art", "8")),
        ("DU/2020/1:par.5", ("par", "5")),
        ("DU/2014/827:art.43A", ("art", "43a")),
    ],
)
def test_unit_of_id(unit_id, expected):
    assert unit_of_id(unit_id) == expected


@pytest.mark.parametrize(
    "provision",
    [
        "art. 27",
        "art. 27 ust. 1 pkt 2",  # a paragraph of a candidate kept whole
        "art. 38",  # an article of which only one paragraph is a candidate
        None,
    ],
)
def test_provision_among_the_candidates_adds_no_warning(provision):
    assert run(payload(matched_provision=provision)).warnings == []


def test_provision_fetched_by_a_tool_counts_as_consulted():
    messages = [
        AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "30"}, "id": "c1"}]),
        tool_result("c1", ["DU/2014/827:art.30"]),
    ]

    result = run(payload(matched_provision="art. 30 ust. 1"), messages)

    assert result.warnings == []
    assert result.consulted_units[-1] == "DU/2014/827:art.30"


@pytest.mark.parametrize(
    "provision",
    [
        "art. 30",  # in the act, but never shown to the agent
        "art. 2",  # must not match art. 27
        "art. 6 ust. 1 lit. b RODO",  # another act, same numbering
        "przepisy ogólne ustawy",  # no article at all
    ],
)
def test_provision_outside_the_consulted_units_adds_a_warning(provision):
    result = run(payload(matched_provision=provision))

    assert result.warnings == [UNCONSULTED_PROVISION_WARNING.format(provision=provision)]
    assert result.confidence == 0.8
    assert result.legal_basis.provision == provision


# Final result (4.4.3)


def test_result_takes_source_metadata_from_code_and_the_assessment_from_the_llm():
    llm_payload = payload(status=ComplianceStatus.NON_COMPLIANT, suggested_fix="Wydłuż termin do 14 dni.")

    result = run(llm_payload)

    assert result.legal_basis.model_dump() == {
        "act_title": ACT.title,
        "eli": ACT.eli,
        "provision": "art. 27",
        "as_of_date": date.today().isoformat(),  # the day of the assessment, not ACT.as_of_date
        "source_url": ACT.source_url,
    }
    assert result.status == ComplianceStatus.NON_COMPLIANT
    assert result.document_excerpt == llm_payload.document_excerpt
    assert result.rationale == llm_payload.rationale
    assert result.suggested_fix == "Wydłuż termin do 14 dni."
    assert result.consulted_units == ["DU/2014/827:art.27", "DU/2014/827:art.38.ust.1"]


def test_both_checks_add_to_the_warnings_collected_earlier():
    earlier = ["Ostrzeżenie ekstrakcji"]

    result = run(payload(document_excerpt="termin 7 dni", matched_provision="art. 30"), warnings=earlier)

    assert result.warnings == [
        "Ostrzeżenie ekstrakcji",
        PARAPHRASED_EXCERPT_WARNING,
        UNCONSULTED_PROVISION_WARNING.format(provision="art. 30"),
    ]
    assert earlier == ["Ostrzeżenie ekstrakcji"]  # the caller's list is not changed
