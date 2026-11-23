import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from legal_checker.agent.state import GraphState
from legal_checker.config import Settings
from legal_checker.models import CheckRequest, ComplianceStatus, EliRef, LlmEvaluationPayload


def test_llm_payload_json_schema():
    schema = LlmEvaluationPayload.model_json_schema()

    assert set(schema["required"]) == {"status", "applicability_unclear", "confidence", "rationale"}
    status_enum = schema["$defs"]["ComplianceStatus"]["enum"]
    assert status_enum == [s.value for s in ComplianceStatus]
    assert schema["properties"]["confidence"]["minimum"] == 0.0
    assert schema["properties"]["confidence"]["maximum"] == 1.0


def test_llm_payload_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        LlmEvaluationPayload(status="ZGODNY", applicability_unclear=False, confidence=1.5, rationale="x")


def test_check_request_min_lengths():
    with pytest.raises(ValidationError):
        CheckRequest(document_fragment="za krótki", act_hint="RODO")
    with pytest.raises(ValidationError):
        CheckRequest(document_fragment="Wystarczająco długi fragment.", act_hint="x")


def test_eli_ref_key():
    assert EliRef(publisher="DU", year=2014, position=827).key == "DU/2014/827"
    assert EliRef.from_key("MP/2023/100") == EliRef(publisher="MP", year=2023, position=100)


def test_settings_defaults():
    settings = Settings()
    assert settings.bm25_top_k == 12
    assert settings.agent_max_tool_calls == 4


def test_graph_state_compiles_and_merges_messages():
    def node(state: GraphState) -> GraphState:
        return {"messages": [AIMessage(content="ok")]}

    builder = StateGraph(GraphState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)

    out = builder.compile().invoke({"messages": [HumanMessage(content="hi")]})

    assert [m.content for m in out["messages"]] == ["hi", "ok"]
