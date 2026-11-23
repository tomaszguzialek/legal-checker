import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from legal_checker.agent import graph as graph_module
from legal_checker.agent.graph import build_graph, check
from legal_checker.agent.nodes import SHORT_CIRCUIT_CONFIDENCE, TOOL_LIMIT_MESSAGE, TOOL_LIMIT_WARNING, EvaluationError
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.config import Settings
from legal_checker.eli_client import EliClient
from legal_checker.models import CheckRequest, ComplianceStatus, LlmEvaluationPayload

from conftest import ELI_TEST_URL

WITHDRAWAL = "Klient może odstąpić od umowy zawartej na odległość w terminie 14 dni bez podania przyczyny."
PAYLOAD = LlmEvaluationPayload(
    status=ComplianceStatus.COMPLIANT,
    applicability_unclear=False,
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


def get_provision(article: str, call_id: str) -> dict:
    return {"name": "get_provision", "args": {"article": article}, "id": call_id}


def research_replies(*tool_calls: list[dict]) -> GenericFakeChatModel:
    """Fake research model: one reply per list of tool calls, then a reply without any."""
    replies = [AIMessage("", tool_calls=calls) for calls in tool_calls]
    return GenericFakeChatModel(messages=iter([*replies, AIMessage("Kontekst jest kompletny.")]))


def evaluation_outputs(*outputs: dict, calls: list | None = None) -> RunnableLambda:
    """Fake evaluation model with ``include_raw=True`` output, returning ``outputs`` in turn."""
    remaining = iter(outputs)
    calls = [] if calls is None else calls
    return RunnableLambda(lambda messages: calls.append(messages) or next(remaining))


def parsed(payload: LlmEvaluationPayload = PAYLOAD) -> dict:
    return {"raw": AIMessage(payload.model_dump_json()), "parsed": payload, "parsing_error": None}


def schema_violation(error: str) -> dict:
    return {"raw": AIMessage('{"status": "PRAWIE ZGODNY"}'), "parsed": None, "parsing_error": ValueError(error)}


def run(eli: EliClient, settings: Settings, research_model, eval_model, fragment: str = WITHDRAWAL) -> dict:
    """Run the graph for a fragment of the consumer rights act on the given (fake) models; return the final state."""
    index = LateBoundBm25Index()
    graph = build_graph(research_model, eval_model, make_tools(index), eli=eli, index=index, settings=settings)
    request = CheckRequest(document_fragment=fragment, act_hint="Dz.U. z 2014 r. poz. 827")
    return graph.invoke({"request": request}, config={"recursion_limit": settings.graph_recursion_limit})


def tool_messages(state: dict) -> list[ToolMessage]:
    return [message for message in state["messages"] if isinstance(message, ToolMessage)]


def test_no_bm25_match_short_circuits_without_calling_a_model(eli, settings):
    state = run(eli, settings, RunnableLambda(unused_model), RunnableLambda(unused_model), "Qwerty zxcvb asdfgh 2099.")

    result = state["result"]
    assert result.status == ComplianceStatus.NOT_APPLICABLE
    assert result.confidence == SHORT_CIRCUIT_CONFIDENCE
    assert result.consulted_units == []
    assert not state["messages"] and "llm_payload" not in state


def test_research_finishing_at_once_goes_straight_to_evaluation(eli, settings):
    eval_calls = []

    state = run(eli, settings, research_replies(), evaluation_outputs(parsed(), calls=eval_calls))

    assert [type(message) for message in state["messages"]] == [AIMessage]
    assert len(eval_calls) == 1
    result = state["result"]
    assert result.status == ComplianceStatus.COMPLIANT
    assert result.legal_basis.provision == "art. 27"
    assert result.consulted_units == [unit.unit_id for unit, _ in state["candidates"]]
    assert result.warnings == ["Ostrzeżenie ekstrakcji"]


def test_provision_fetched_with_get_provision_is_consulted_and_evaluated(eli, settings):
    eval_calls = []

    research = research_replies([get_provision("30", "c1")])

    state = run(eli, settings, research, evaluation_outputs(parsed(), calls=eval_calls))

    assert "DU/2014/827:art.30" not in [unit.unit_id for unit, _ in state["candidates"]]
    [result_message] = tool_messages(state)
    assert result_message.artifact == ["DU/2014/827:art.30"]
    assert state["result"].consulted_units[-1] == "DU/2014/827:art.30"
    assert "<tool_results>\n[DU/2014/827:art.30]" in eval_calls[0][1].content
    assert TOOL_LIMIT_WARNING not in state["result"].warnings


def test_tool_call_budget_breaks_the_research_loop_with_a_warning(eli, settings):
    settings = settings.model_copy(update={"agent_max_tool_calls": 2})
    # Asks for one more provision on every turn, far beyond the budget; the fake would then finish.
    research = research_replies(*[[get_provision(str(article), f"c{article}")] for article in range(1, 10)])

    state = run(eli, settings, research, evaluation_outputs(parsed()))

    assert [message.artifact for message in tool_messages(state)] == [["DU/2014/827:art.1"], ["DU/2014/827:art.2"]]
    assert state["messages"][-1].tool_calls == [{**get_provision("3", "c3"), "type": "tool_call"}]  # requested, never run
    assert state["result"].warnings == ["Ostrzeżenie ekstrakcji", TOOL_LIMIT_WARNING]
    assert state["result"].consulted_units[-2:] == ["DU/2014/827:art.1", "DU/2014/827:art.2"]


def test_calls_of_one_reply_beyond_the_budget_are_refused(eli, settings):
    settings = settings.model_copy(update={"agent_max_tool_calls": 2})
    research = research_replies([get_provision("1", "c1"), get_provision("2", "c2"), get_provision("3", "c3")])

    state = run(eli, settings, research, evaluation_outputs(parsed()))

    messages = tool_messages(state)
    assert [message.artifact for message in messages] == [["DU/2014/827:art.1"], ["DU/2014/827:art.2"], []]
    assert messages[2].content == TOOL_LIMIT_MESSAGE
    assert isinstance(state["messages"][-1], AIMessage) and not state["messages"][-1].tool_calls
    assert TOOL_LIMIT_WARNING in state["result"].warnings
    assert "DU/2014/827:art.3" not in state["result"].consulted_units


def test_schema_violation_is_retried_once(eli, settings):
    eval_calls = []

    eval_model = evaluation_outputs(schema_violation("status: invalid enum"), parsed(), calls=eval_calls)

    state = run(eli, settings, research_replies(), eval_model)

    assert state["schema_errors"] == ["status: invalid enum"]
    assert state["eval_attempts"] == 2
    assert "status: invalid enum" in eval_calls[1][-1].content
    assert state["result"].status == ComplianceStatus.COMPLIANT


def test_second_schema_violation_ends_the_graph_with_evaluation_error(eli, settings):
    eval_model = evaluation_outputs(schema_violation("first"), schema_violation("second"))

    with pytest.raises(EvaluationError, match="ostatni błąd: second"):
        run(eli, settings, research_replies(), eval_model)
