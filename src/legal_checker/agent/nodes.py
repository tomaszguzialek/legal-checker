"""Graph nodes: deterministic steps and LLM calls (SPEC Task 4.3).

Each node takes the graph state and returns a partial update of it.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from legal_checker import legal_chunker
from legal_checker.agent.prompts import RESEARCH_SYSTEM_PROMPT
from legal_checker.agent.state import GraphState
from legal_checker.agent.tools import format_unit
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.eli_client import EliClient
from legal_checker.eli_resolver import resolve_hint
from legal_checker.models import ActDocument, ComplianceResult, ComplianceStatus, LlmEvaluationPayload
from legal_checker.post_validator import post_validate
from legal_checker.text_extractor import extract_act_text

Node = Callable[[GraphState], dict]

# No act token occurs in the fragment at all, but a provision could still apply in other words.
SHORT_CIRCUIT_CONFIDENCE = 0.9
SHORT_CIRCUIT_RATIONALE = (
    "Żaden przepis aktu nie zawiera słów kluczowych występujących we fragmencie dokumentu. "
    "Akt nie daje więc podstaw do oceny tego fragmentu. "
    "Wynik ustalono bez wywołania modelu językowego."
)


def make_act_nodes(eli: EliClient, index: LateBoundBm25Index, top_k: int) -> dict[str, Node]:
    """Nodes that resolve, fetch, chunk and search the act of one request, keyed by node name.

    ``retrieve`` builds the request's index in ``index``, which the agent tools search later.
    """

    def resolve_act(state: GraphState) -> dict:
        return {"eli_ref": resolve_hint(state["request"].act_hint, eli.http)}

    def fetch_act(state: GraphState) -> dict:
        ref = state["eli_ref"]
        fetched = eli.fetch_act(ref)
        extracted = extract_act_text(ref, fetched.content, eli.cache)
        act = ActDocument(
            eli=ref.key,
            title=fetched.metadata.title,
            source_url=fetched.content.url,
            as_of_date=fetched.fetched_at.date().isoformat(),
            text=extracted.text,
        )
        return {"act": act, "warnings": extracted.warnings}

    def chunk_act(state: GraphState) -> dict:
        chunked = legal_chunker.chunk_act(state["act"].eli, state["act"].text)
        return {"units": chunked.units, "warnings": chunked.warnings}

    def retrieve(state: GraphState) -> dict:
        built = index.build(state["units"])
        return {"candidates": built.search(state["request"].document_fragment, top_k=top_k)}

    return {
        "resolve_act": resolve_act,
        "fetch_act": fetch_act,
        "chunk_act": chunk_act,
        "retrieve": retrieve,
    }


def route_after_retrieve(state: GraphState) -> Literal["short_circuit_na", "research"]:
    """Skip the LLM when no unit shares a single token with the fragment (max BM25 score 0)."""
    if max((score for _, score in state["candidates"]), default=0.0) <= 0.0:
        return "short_circuit_na"
    return "research"


def short_circuit_na(state: GraphState) -> dict:
    payload = LlmEvaluationPayload(
        status=ComplianceStatus.NOT_APPLICABLE,
        confidence=SHORT_CIRCUIT_CONFIDENCE,
        rationale=SHORT_CIRCUIT_RATIONALE,
    )
    return {"result": _post_validate(payload, state)}


def make_research_node(research_model: Runnable) -> Node:
    """The agent step: the model reviews the candidates and either calls tools or finishes.

    ``messages`` holds only the research conversation (the model's ``AIMessage``s and the
    ``ToolMessage``s answering them); the system prompt and the fragment with its candidates
    are put in front of it on every call, so they are built from the state the same way each time.
    """

    def research(state: GraphState) -> dict:
        prompt = [SystemMessage(RESEARCH_SYSTEM_PROMPT), HumanMessage(context_prompt(state))]
        return {"messages": [research_model.invoke([*prompt, *state.get("messages", [])])]}

    return research


def context_prompt(state: GraphState) -> str:
    """The fragment under review and the BM25 candidates, in the tagged sections the prompts refer to."""
    provisions = "\n\n".join(format_unit(unit) for unit, _ in state["candidates"])
    return (
        f"<document_fragment>\n{state['request'].document_fragment}\n</document_fragment>\n\n"
        f"<legal_provisions>\n{provisions}\n</legal_provisions>"
    )


def validate(state: GraphState) -> dict:
    return {"result": _post_validate(state["llm_payload"], state)}


def _post_validate(payload: LlmEvaluationPayload, state: GraphState) -> ComplianceResult:
    return post_validate(
        payload,
        act=state["act"],
        candidates=state["candidates"],
        messages=state.get("messages", []),
        warnings=state.get("warnings", []),
    )
