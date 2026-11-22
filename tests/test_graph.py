import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from legal_checker.agent import graph as graph_module
from legal_checker.agent.graph import build_graph, check
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.config import Settings
from legal_checker.models import CheckRequest, ComplianceStatus, LlmEvaluationPayload

from conftest import ELI_TEST_URL

WITHDRAWAL = "Klient może odstąpić od umowy zawartej na odległość w terminie 14 dni bez podania przyczyny."
PAYLOAD = LlmEvaluationPayload(
    status=ComplianceStatus.COMPLIANT,
    confidence=0.9,
    document_excerpt="w terminie 14 dni",
    matched_provision="art. 27",
    rationale="Zapis powtarza ustawowy termin odstąpienia.",
)


def unused_model(_):
    raise AssertionError("model must not be called")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(eli_api_base_url=ELI_TEST_URL, cache_dir=tmp_path)


def test_topology_matches_spec_section_3_5(eli, settings):
    index = LateBoundBm25Index()
    graph = build_graph(
        RunnableLambda(unused_model),
        RunnableLambda(unused_model),
        make_tools(index),
        eli=eli,
        index=index,
        settings=settings,
    ).get_graph()

    edges = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}

    assert edges == {
        ("__start__", "resolve_act", False),
        ("resolve_act", "fetch_act", False),
        ("fetch_act", "chunk_act", False),
        ("chunk_act", "retrieve", False),
        ("retrieve", "short_circuit_na", True),
        ("retrieve", "research", True),
        ("short_circuit_na", "__end__", False),
        ("research", "tools", True),
        ("research", "evaluate", True),
        ("tools", "research", False),
        ("evaluate", "evaluate", True),
        ("evaluate", "validate", True),
        ("validate", "__end__", False),
    }
    assert "evaluate -.-> evaluate;" in graph.draw_mermaid()


def test_check_runs_the_graph_with_the_models_built_for_the_request(cached_act, settings, monkeypatch):
    built = {}

    def fake_build_models(model_settings, tools):
        built.update(settings=model_settings, tools=[tool.name for tool in tools])
        research = RunnableLambda(lambda _: AIMessage("Kontekst jest kompletny."))
        evaluation = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": PAYLOAD, "parsing_error": None})
        return research, evaluation

    monkeypatch.setattr(graph_module, "build_models", fake_build_models)

    result = check(CheckRequest(document_fragment=WITHDRAWAL, act_hint="DU/2014/827"), settings)

    assert built == {"settings": settings, "tools": ["search_provisions", "get_provision"]}
    assert result.status == ComplianceStatus.COMPLIANT
    assert result.legal_basis.eli == "DU/2014/827"
    assert result.legal_basis.source_url == cached_act.content.url
    assert "DU/2014/827:art.27" in result.consulted_units
    assert result.warnings == ["Ostrzeżenie ekstrakcji"]
