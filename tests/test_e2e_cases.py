"""End-to-end cases of the compliance check (SPEC Task 5.2).

Six scenarios covering every assessment category and the edge cases. By default each one runs
the whole graph on the cached consumer rights act and fake models, verifying the path through
the graph rather than the content of the assessment.
"""

from dataclasses import dataclass

import pytest
import respx
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from legal_checker.agent import graph as graph_module
from legal_checker.config import Settings
from legal_checker.models import CheckRequest, ComplianceStatus, LlmEvaluationPayload

from conftest import ELI_TEST_URL


@dataclass(frozen=True)
class Case:
    id: str
    fragment: str
    hint: str
    expected_status: ComplianceStatus | None  # None: any status, the case is about how sure the model is
    provision: str | None  # the article the assessment hinges on; None when the act does not apply
    max_confidence: float = 1.0


WITHDRAWAL_ELI = "DU/2014/827"
WITHDRAWAL_SIGNATURE = "Dz.U. z 2014 r. poz. 827"

COMPLIANT = Case(
    id="zgodny",
    fragment=(
        "Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić bez podawania "
        "przyczyny i bez ponoszenia kosztów, z wyjątkiem kosztów określonych w art. 33, art. 34 ust. 2 "
        "i art. 35 ustawy o prawach konsumenta."
    ),
    hint=WITHDRAWAL_ELI,
    expected_status=ComplianceStatus.COMPLIANT,
    provision="27",
)
PARTIALLY_COMPLIANT = Case(
    id="czesciowo-zgodny",
    # Says how to withdraw, but not by when.
    fragment=(
        "Przysługuje Państwu prawo odstąpienia od umowy zawartej na odległość bez podawania przyczyny. "
        "Aby z niego skorzystać, należy przesłać nam oświadczenie o odstąpieniu od umowy, "
        "np. pocztą elektroniczną na adres zwroty@sklep.example."
    ),
    hint=WITHDRAWAL_SIGNATURE,
    expected_status=ComplianceStatus.PARTIALLY_COMPLIANT,
    provision="27",
)
NON_COMPLIANT = Case(
    id="niezgodny",
    fragment=(
        "Klient może odstąpić od umowy zawartej na odległość w terminie 7 dni od dnia otrzymania towaru. "
        "Po upływie tego terminu zwroty nie są przyjmowane."
    ),
    hint=WITHDRAWAL_SIGNATURE,
    expected_status=ComplianceStatus.NON_COMPLIANT,
    provision="27",
)
NOT_APPLICABLE = Case(
    id="nie-dotyczy",
    fragment=(
        "Cotygodniowe spotkania zespołu odbywają się we wtorki o godzinie 10:00 "
        "w sali konferencyjnej na drugim piętrze."
    ),
    hint="ustawa o prawach konsumenta",
    expected_status=ComplianceStatus.NOT_APPLICABLE,
    provision=None,
)
AMBIGUOUS = Case(
    id="graniczny",
    # Whether the parties are businesses or a business and a consumer is left open.
    fragment=(
        "Strona odstępująca od umowy bez ważnej przyczyny zapłaci drugiej stronie karę umowną "
        "w wysokości 20% wartości zamówienia."
    ),
    hint=WITHDRAWAL_SIGNATURE,
    expected_status=None,
    provision="27",
    max_confidence=0.7,
)
CATEGORY_CASES = [COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE, AMBIGUOUS]

# Case 6: the compliant case for each kind of hint. The link is only parsed, never requested.
HINT_FORMATS = {
    "link-eli": "https://eli.gov.pl/eli/DU/2014/827/ogl",
    "sygnatura": WITHDRAWAL_SIGNATURE,
    "tytul": "ustawa o prawach konsumenta",
}

# Trimmed from a real /acts/search?title=ustawa o prawach konsumenta response.
TITLE_SEARCH_ITEMS = [
    {
        "ELI": "DU/2022/2581",
        "publisher": "DU",
        "year": 2022,
        "pos": 2581,
        "title": "Ustawa z dnia 1 grudnia 2022 r. o zmianie ustawy o prawach konsumenta oraz niektórych innych ustaw",
        "type": "Ustawa",
        "inForce": "IN_FORCE",
    },
    {
        "ELI": "DU/2014/827",
        "publisher": "DU",
        "year": 2014,
        "pos": 827,
        "title": "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        "type": "Ustawa",
        "inForce": "IN_FORCE",
    },
]

ACT_NODES = ["resolve_act", "fetch_act", "chunk_act", "retrieve"]


def fake_payload(case: Case) -> LlmEvaluationPayload:
    """A canned assessment in the case's category; the fake run checks the graph, not this content."""
    return LlmEvaluationPayload(
        status=case.expected_status or ComplianceStatus.PARTIALLY_COMPLIANT,
        confidence=min(0.9, case.max_confidence),
        document_excerpt=case.fragment.split(".")[0] if case.provision else None,
        matched_provision=f"art. {case.provision}" if case.provision else None,
        rationale="Ocena testowa.",
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(eli_api_base_url=ELI_TEST_URL, cache_dir=tmp_path)


@pytest.fixture
def title_search():
    with respx.mock(base_url=ELI_TEST_URL, assert_all_called=False) as mock:
        yield mock.get("/acts/search").respond(json={"count": len(TITLE_SEARCH_ITEMS), "items": TITLE_SEARCH_ITEMS})


def run_on_fake_models(case: Case, hint: str, settings: Settings, monkeypatch) -> tuple[dict, list[str]]:
    """Run the graph on fake models for the case; return the final state and the nodes in the order they ran.

    The fake research model reads the case's provision with ``get_provision`` (if it has one) and then finishes.
    """

    def fake_build_models(_settings, _tools):
        calls = [{"name": "get_provision", "args": {"article": case.provision}, "id": "c1"}] if case.provision else []
        replies = [AIMessage("", tool_calls=calls)] if calls else []
        research = GenericFakeChatModel(messages=iter([*replies, AIMessage("Kontekst jest kompletny.")]))
        payload = fake_payload(case)
        evaluation = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": payload, "parsing_error": None})
        return research, evaluation

    monkeypatch.setattr(graph_module, "build_models", fake_build_models)
    steps = []
    request = CheckRequest(document_fragment=case.fragment, act_hint=hint)
    state = graph_module.run(request, settings, on_step=lambda node, _: steps.append(node))
    return state, steps


@pytest.mark.parametrize("case", CATEGORY_CASES, ids=lambda case: case.id)
def test_case_runs_through_research_and_evaluation(case, cached_act, title_search, settings, monkeypatch):
    state, steps = run_on_fake_models(case, case.hint, settings, monkeypatch)

    research = ["research", "tools", "research"] if case.provision else ["research"]
    assert steps == [*ACT_NODES, *research, "evaluate", "validate"]
    assert state["eli_ref"].key == "DU/2014/827"
    assert state["candidates"], "BM25 found no provision sharing a word with the fragment"
    result = state["result"]
    assert result.status == fake_payload(case).status
    assert result.legal_basis.eli == "DU/2014/827"
    assert result.legal_basis.source_url == cached_act.content.url
    if case.provision:
        assert f"DU/2014/827:art.{case.provision}" in result.consulted_units
    assert result.warnings == ["Ostrzeżenie ekstrakcji"]


@pytest.mark.parametrize("hint", HINT_FORMATS.values(), ids=HINT_FORMATS.keys())
def test_every_hint_format_resolves_to_the_same_act(hint, cached_act, title_search, settings, monkeypatch):
    state, steps = run_on_fake_models(COMPLIANT, hint, settings, monkeypatch)

    assert steps[: len(ACT_NODES)] == ACT_NODES
    assert state["eli_ref"].key == "DU/2014/827"
    assert state["result"].legal_basis.eli == "DU/2014/827"
    assert title_search.called == (hint == HINT_FORMATS["tytul"])
