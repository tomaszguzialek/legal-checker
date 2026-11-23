"""Graph nodes: deterministic steps and LLM calls (SPEC Task 4.3).

Each node takes the graph state and returns a partial update of it.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

from legal_checker import legal_chunker
from legal_checker.agent.prompts import EVALUATION_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from legal_checker.agent.state import GraphState
from legal_checker.agent.tools import format_unit
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.eli_client import EliClient
from legal_checker.eli_resolver import resolve_hint
from legal_checker.models import ActDocument, ComplianceResult, ComplianceStatus, LlmEvaluationPayload
from legal_checker.post_validator import post_validate
from legal_checker.text_extractor import extract_act_text

Node = Callable[[GraphState], dict]

# Attempts of one LLM call that fails with a provider error (rate limit, timeout, 5xx).
PROVIDER_ATTEMPTS = 2
# Evaluations in a row that may return a payload violating the schema: the first is retried once.
MAX_EVAL_ATTEMPTS = 2

# No act token occurs in the fragment at all, but a provision could still apply in other words.
SHORT_CIRCUIT_CONFIDENCE = 0.9
SHORT_CIRCUIT_RATIONALE = (
    "Żaden przepis aktu nie zawiera słów kluczowych występujących we fragmencie dokumentu. "
    "Akt nie daje więc podstaw do oceny tego fragmentu. "
    "Wynik ustalono bez wywołania modelu językowego."
)
TOOL_LIMIT_WARNING = "Osiągnięto limit wywołań narzędzi agenta"
SCHEMA_RETRY_MESSAGE = (
    "Poprzednia odpowiedź nie spełniła wymaganego schematu JSON: {error}\n"
    "Odpowiedz ponownie, ściśle według schematu."
)
TOOL_LIMIT_MESSAGE = "Nie wykonano: osiągnięto limit wywołań narzędzi. Oceń zgodność na podstawie zebranego kontekstu."


class EvaluationError(RuntimeError):
    """The evaluation model failed to return a payload matching the schema, even when retried."""


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
        applicability_unclear=False,
        confidence=SHORT_CIRCUIT_CONFIDENCE,
        rationale=SHORT_CIRCUIT_RATIONALE,
    )
    return {"result": _post_validate(payload, state)}


def make_research_node(research_model: Runnable, max_tool_calls: int) -> Node:
    """The agent step: the model reviews the candidates and either calls tools or finishes.

    ``messages`` holds only the research conversation (the model's ``AIMessage``s and the
    ``ToolMessage``s answering them); the system prompt and the fragment with its candidates
    are put in front of it on every call, so they are built from the state the same way each time.
    A reply asking for more tool calls than the budget has left adds ``TOOL_LIMIT_WARNING``.
    """

    model = research_model.with_retry(stop_after_attempt=PROVIDER_ATTEMPTS)

    def research(state: GraphState) -> dict:
        prompt = [SystemMessage(RESEARCH_SYSTEM_PROMPT), HumanMessage(context_prompt(state))]
        reply = model.invoke([*prompt, *state.get("messages", [])])
        update: dict = {"messages": [reply]}
        if len(reply.tool_calls) > remaining_tool_calls(state, max_tool_calls):
            update["warnings"] = [TOOL_LIMIT_WARNING]
        return update

    return research


def make_route_after_research(max_tool_calls: int) -> Callable[[GraphState], Literal["tools", "evaluate"]]:
    """Run the tools while the last reply asks for any and the budget is not used up."""

    def route_after_research(state: GraphState) -> Literal["tools", "evaluate"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and remaining_tool_calls(state, max_tool_calls) > 0:
            return "tools"
        return "evaluate"

    return route_after_research


def make_tools_node(tools: list[BaseTool], max_tool_calls: int) -> ToolNode:
    """``ToolNode`` running the last reply's tool calls in order, as far as the budget allows.

    Calls beyond it are not run: each gets a ``ToolMessage`` saying so (with no artifact), so the
    conversation stays well-formed for the provider when the research model is called again.
    """

    def within_budget(request: ToolCallRequest, execute: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
        state = request.state
        position = [call["id"] for call in state["messages"][-1].tool_calls].index(request.tool_call["id"])
        if position < remaining_tool_calls(state, max_tool_calls):
            return execute(request)
        return ToolMessage(
            TOOL_LIMIT_MESSAGE,
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            artifact=[],
            status="error",
        )

    return ToolNode(tools, wrap_tool_call=within_budget)


def remaining_tool_calls(state: GraphState, max_tool_calls: int) -> int:
    used = sum(isinstance(message, ToolMessage) for message in state.get("messages", []))
    return max(max_tool_calls - used, 0)


def context_prompt(state: GraphState) -> str:
    """The fragment under review and the BM25 candidates, in the tagged sections the prompts refer to."""
    provisions = "\n\n".join(format_unit(unit) for unit, _ in state["candidates"])
    return (
        f"<document_fragment>\n{state['request'].document_fragment}\n</document_fragment>\n\n"
        f"<legal_provisions>\n{provisions}\n</legal_provisions>"
    )


def make_evaluate_node(eval_model: Runnable) -> Node:
    """The assessment: the structured-output model judges the fragment against all provisions gathered.

    ``eval_model`` returns ``{"raw", "parsed", "parsing_error"}``. A schema violation is recorded in
    ``schema_errors`` and the evaluation runs again with the error appended, at most
    ``MAX_EVAL_ATTEMPTS`` times in total; then ``EvaluationError`` ends the graph.
    """
    model = eval_model.with_retry(stop_after_attempt=PROVIDER_ATTEMPTS)

    def evaluate(state: GraphState) -> dict:
        errors = state.get("schema_errors", [])
        messages = [SystemMessage(EVALUATION_SYSTEM_PROMPT), HumanMessage(evaluation_prompt(state))]
        if errors:
            messages.append(HumanMessage(SCHEMA_RETRY_MESSAGE.format(error=errors[-1])))
        output = model.invoke(messages)
        attempts = state.get("eval_attempts", 0) + 1
        if output["parsing_error"] is None and output["parsed"] is not None:
            return {"llm_payload": output["parsed"], "eval_attempts": attempts}
        error = str(output["parsing_error"] or "Brak sparsowanej odpowiedzi")
        if attempts >= MAX_EVAL_ATTEMPTS:
            raise EvaluationError(
                f"Model oceny {attempts} razy zwrócił odpowiedź niezgodną ze schematem; ostatni błąd: {error}"
            )
        return {"llm_payload": None, "schema_errors": [*errors, error], "eval_attempts": attempts}

    return evaluate


def route_after_evaluate(state: GraphState) -> Literal["evaluate", "validate"]:
    """Retry an evaluation that violated the schema; ``evaluate`` raises once out of attempts."""
    return "validate" if state.get("llm_payload") is not None else "evaluate"


def evaluation_prompt(state: GraphState) -> str:
    """The research context plus the provisions the tools returned, in a ``<tool_results>`` section.

    Only results that brought provisions are included: "not found" and over-budget replies are noise here.
    """
    results = [
        message.content
        for message in state.get("messages", [])
        if isinstance(message, ToolMessage) and message.artifact
    ]
    if not results:
        return context_prompt(state)
    return context_prompt(state) + "\n\n<tool_results>\n" + "\n\n".join(results) + "\n</tool_results>"


def validate(state: GraphState) -> dict:
    return {"result": _post_validate(state["llm_payload"], state)}


def _post_validate(payload: LlmEvaluationPayload, state: GraphState) -> ComplianceResult:
    return post_validate(
        payload,
        document_fragment=state["request"].document_fragment,
        act=state["act"],
        candidates=state["candidates"],
        messages=state.get("messages", []),
        warnings=state.get("warnings", []),
    )
