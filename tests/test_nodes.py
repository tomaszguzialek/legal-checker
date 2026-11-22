import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from legal_checker.agent.nodes import (
    SHORT_CIRCUIT_CONFIDENCE,
    TOOL_LIMIT_MESSAGE,
    TOOL_LIMIT_WARNING,
    EvaluationError,
    make_act_nodes,
    make_evaluate_node,
    make_research_node,
    make_route_after_research,
    make_tools_node,
    route_after_evaluate,
    route_after_retrieve,
    short_circuit_na,
    validate,
)
from legal_checker.agent.prompts import EVALUATION_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from legal_checker.agent.state import GraphState
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.eli_client import EliClient
from legal_checker.models import (
    ActDocument,
    CheckRequest,
    ComplianceStatus,
    LegalUnit,
    LlmEvaluationPayload,
)

from conftest import CONSUMER_RIGHTS, CONSUMER_RIGHTS_TEXT

WITHDRAWAL = "Klient może odstąpić od umowy zawartej na odległość w terminie 14 dni bez podania przyczyny."


def run_act_nodes(eli: EliClient, fragment: str, index: LateBoundBm25Index | None = None) -> dict:
    """Run the deterministic nodes in graph order, merging updates (warnings are appended)."""
    nodes = make_act_nodes(eli, index or LateBoundBm25Index(), top_k=5)
    state: dict = {"request": CheckRequest(document_fragment=fragment, act_hint="Dz.U. z 2014 r. poz. 827")}
    for name in ["resolve_act", "fetch_act", "chunk_act", "retrieve"]:
        update = nodes[name](state)
        state = {**state, **update, "warnings": state.get("warnings", []) + update.get("warnings", [])}
    return state


def test_act_nodes_resolve_fetch_chunk_and_retrieve(eli, cached_act):
    index = LateBoundBm25Index()

    state = run_act_nodes(eli, WITHDRAWAL, index)

    assert state["eli_ref"] == CONSUMER_RIGHTS
    assert state["act"] == ActDocument(
        eli="DU/2014/827",
        title="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        source_url=cached_act.content.url,
        as_of_date=cached_act.fetched_at.date().isoformat(),
        text=CONSUMER_RIGHTS_TEXT,
    )
    assert state["units"] and all(unit.eli == "DU/2014/827" for unit in state["units"])
    assert state["warnings"] == ["Ostrzeżenie ekstrakcji"]
    assert "DU/2014/827:art.27" in [unit.unit_id for unit, _ in state["candidates"]]
    assert index.get("27").unit_id == "DU/2014/827:art.27"  # the tools' index is built
    assert route_after_retrieve(state) == "research"


def test_late_bound_index_fails_clearly_before_retrieve():
    with pytest.raises(RuntimeError, match="not built yet"):
        LateBoundBm25Index().search("odstąpienie")


def test_no_shared_token_short_circuits_to_not_applicable(eli, cached_act):
    state = run_act_nodes(eli, "Qwerty zxcvb asdfgh 2099.")

    assert state["candidates"] == []
    assert route_after_retrieve(state) == "short_circuit_na"

    result = short_circuit_na(state)["result"]

    assert result.status == ComplianceStatus.NOT_APPLICABLE
    assert result.confidence == SHORT_CIRCUIT_CONFIDENCE
    assert result.document_excerpt is None and result.legal_basis.provision is None
    assert result.legal_basis.eli == "DU/2014/827"
    assert result.legal_basis.source_url == cached_act.content.url
    assert result.consulted_units == []
    assert result.warnings == ["Ostrzeżenie ekstrakcji"]


def test_validate_attaches_metadata_and_everything_the_agent_saw(eli, cached_act):
    state = run_act_nodes(eli, WITHDRAWAL)
    unit = LegalUnit(unit_id="DU/2014/827:art.38", eli="DU/2014/827", article="38", hierarchy_path=[], text="Art. 38.")
    candidate_ids = [unit.unit_id for unit, _ in state["candidates"]]
    state["messages"] = [
        AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "38"}, "id": "c1"}]),
        ToolMessage("[DU/2014/827:art.38] Art. 38.", tool_call_id="c1", artifact=[unit.unit_id, candidate_ids[0]]),
    ]
    state["llm_payload"] = LlmEvaluationPayload(
        status=ComplianceStatus.COMPLIANT,
        confidence=0.85,
        document_excerpt="w terminie 14 dni",
        matched_provision="art. 27",
        rationale="Zapis powtarza ustawowy termin.",
    )

    result = validate(state)["result"]

    assert result.status == ComplianceStatus.COMPLIANT and result.confidence == 0.85
    assert result.legal_basis.model_dump() == {
        "act_title": "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        "eli": "DU/2014/827",
        "provision": "art. 27",
        "as_of_date": cached_act.fetched_at.date().isoformat(),
        "source_url": cached_act.content.url,
    }
    assert result.consulted_units == [*candidate_ids, "DU/2014/827:art.38"]


def test_research_sends_prompt_fragment_and_candidates_then_appends_the_reply(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    reply = AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "38"}, "id": "c1"}])
    calls = []
    research = make_research_node(RunnableLambda(lambda messages: calls.append(messages) or reply), max_tool_calls=4)

    update = research(state)

    assert update == {"messages": [reply]}
    system, human = calls[0]
    assert isinstance(system, SystemMessage) and system.content == RESEARCH_SYSTEM_PROMPT
    assert isinstance(human, HumanMessage)
    assert human.content.startswith(f"<document_fragment>\n{WITHDRAWAL}\n</document_fragment>\n\n<legal_provisions>\n[")
    assert human.content.endswith("</legal_provisions>")
    for unit, _ in state["candidates"]:
        assert f"[{unit.unit_id}] {' > '.join(unit.hierarchy_path)}\n{unit.text}" in human.content


def test_research_continues_the_conversation_after_tool_results(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    state["messages"] = [
        AIMessage("", tool_calls=[{"name": "get_provision", "args": {"article": "38"}, "id": "c1"}]),
        ToolMessage("[DU/2014/827:art.38] Art. 38.", tool_call_id="c1", artifact=["DU/2014/827:art.38"]),
    ]
    model = GenericFakeChatModel(messages=iter([AIMessage("Kontekst jest kompletny.")]))
    calls = []
    research = make_research_node(
        RunnableLambda(lambda messages: calls.append(messages) or model.invoke(messages)), max_tool_calls=4
    )

    reply = research(state)["messages"][0]

    assert reply.content == "Kontekst jest kompletny." and not reply.tool_calls
    assert [type(m) for m in calls[0]] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert calls[0][2:] == state["messages"]


def get_provision_call(article: str, call_id: str) -> dict:
    return {"name": "get_provision", "args": {"article": article}, "id": call_id}


def answered(*articles: str) -> list:
    """A reply asking for ``get_provision`` of each article, followed by the tool results."""
    calls = [get_provision_call(article, f"done-{article}") for article in articles]
    results = [
        ToolMessage(f"[DU/2014/827:art.{a}]", tool_call_id=f"done-{a}", artifact=[f"DU/2014/827:art.{a}"]) for a in articles
    ]
    return [AIMessage("", tool_calls=calls), *results]


@pytest.mark.parametrize(
    ("history", "tool_calls", "expected"),
    [
        ([], [get_provision_call("38", "c1")], "tools"),
        ([], [], "evaluate"),
        (answered("27"), [get_provision_call("38", "c1")], "tools"),
        (answered("27", "28"), [get_provision_call("38", "c1")], "evaluate"),  # budget of 2 used up
    ],
)
def test_route_after_research(history, tool_calls, expected):
    state = {"messages": [*history, AIMessage("", tool_calls=tool_calls)]}

    assert make_route_after_research(max_tool_calls=2)(state) == expected


@pytest.mark.parametrize(("history", "requested", "warned"), [([], 2, False), ([], 3, True), (answered("27", "28"), 1, True)])
def test_research_warns_when_the_reply_asks_for_more_calls_than_the_budget_left(eli, history, requested, warned):
    state = {**run_act_nodes(eli, WITHDRAWAL), "messages": history}
    reply = AIMessage("", tool_calls=[get_provision_call(str(30 + i), f"c{i}") for i in range(requested)])

    update = make_research_node(RunnableLambda(lambda _: reply), max_tool_calls=2)(state)

    assert update.get("warnings") == ([TOOL_LIMIT_WARNING] if warned else None)


def test_tools_node_runs_calls_in_order_only_up_to_the_budget(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    index = LateBoundBm25Index()
    index.build(state["units"])
    graph = StateGraph(GraphState)
    graph.add_node("tools", make_tools_node(make_tools(index), max_tool_calls=3))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    reply = AIMessage("", tool_calls=[get_provision_call(article, f"c{article}") for article in ["38", "34", "35"]])

    messages = graph.compile().invoke({"messages": [*answered("27"), reply]})["messages"][3:]

    assert [(m.tool_call_id, m.artifact) for m in messages] == [
        ("c38", ["DU/2014/827:art.38"]),
        ("c34", ["DU/2014/827:art.34"]),
        ("c35", []),
    ]
    assert messages[0].content.startswith("[DU/2014/827:art.38]")
    assert messages[2].content == TOOL_LIMIT_MESSAGE and messages[2].status == "error"


PAYLOAD = LlmEvaluationPayload(
    status=ComplianceStatus.COMPLIANT,
    confidence=0.9,
    document_excerpt="w terminie 14 dni",
    matched_provision="art. 27",
    rationale="Zapis powtarza ustawowy termin odstąpienia.",
)


def parsed(payload: LlmEvaluationPayload = PAYLOAD) -> dict:
    return {"raw": AIMessage(payload.model_dump_json()), "parsed": payload, "parsing_error": None}


def schema_violation(error: str = "confidence: Input should be less than or equal to 1") -> dict:
    return {"raw": AIMessage('{"confidence": 7}'), "parsed": None, "parsing_error": ValueError(error)}


def scripted(*outputs, calls: list | None = None) -> RunnableLambda:
    """Evaluation model returning ``outputs`` in turn (raising those that are exceptions), recording its inputs."""
    remaining = iter(outputs)
    calls = [] if calls is None else calls

    def respond(messages):
        calls.append(messages)
        output = next(remaining)
        if isinstance(output, Exception):
            raise output
        return output

    return RunnableLambda(respond)


def test_evaluate_sends_fragment_candidates_and_the_provisions_tools_returned(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    refused = ToolMessage(TOOL_LIMIT_MESSAGE, tool_call_id="c9", artifact=[], status="error")
    state["messages"] = [*answered("38"), refused]
    calls = []

    update = make_evaluate_node(scripted(parsed(), calls=calls))(state)

    assert update == {"llm_payload": PAYLOAD, "eval_attempts": 1}
    assert route_after_evaluate({**state, **update}) == "validate"
    system, human = calls[0]
    assert system.content == EVALUATION_SYSTEM_PROMPT
    assert "</legal_provisions>\n\n<tool_results>\n[DU/2014/827:art.38]\n</tool_results>" in human.content
    assert TOOL_LIMIT_MESSAGE not in human.content


def test_evaluate_without_tool_results_has_no_tool_results_section(eli):
    calls = []

    make_evaluate_node(scripted(parsed(), calls=calls))(run_act_nodes(eli, WITHDRAWAL))

    assert calls[0][1].content.endswith("</legal_provisions>")


def test_evaluate_retries_a_schema_violation_once_with_the_error_attached(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    calls = []
    evaluate = make_evaluate_node(scripted(schema_violation("confidence too high"), parsed(), calls=calls))

    first = evaluate(state)
    assert first == {"llm_payload": None, "schema_errors": ["confidence too high"], "eval_attempts": 1}
    state = {**state, **first}
    assert route_after_evaluate(state) == "evaluate"

    second = evaluate(state)

    assert second == {"llm_payload": PAYLOAD, "eval_attempts": 2}
    assert len(calls[1]) == 3 and calls[1][:2] == calls[0]
    assert calls[1][2].content.startswith("Poprzednia odpowiedź nie spełniła wymaganego schematu JSON: confidence too high")


def test_evaluate_raises_evaluation_error_after_the_second_schema_violation(eli):
    state = run_act_nodes(eli, WITHDRAWAL)
    evaluate = make_evaluate_node(scripted(schema_violation("first"), schema_violation("second")))
    state = {**state, **evaluate(state)}

    with pytest.raises(EvaluationError, match="2 razy.*ostatni błąd: second"):
        evaluate(state)


def test_evaluate_retries_a_provider_error_once(eli):
    calls = []

    update = make_evaluate_node(scripted(ConnectionError("503"), parsed(), calls=calls))(run_act_nodes(eli, WITHDRAWAL))

    assert update["llm_payload"] == PAYLOAD and update["eval_attempts"] == 1
    assert len(calls) == 2
