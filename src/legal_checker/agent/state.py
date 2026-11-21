"""LangGraph state (SPEC section 3.4)."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from legal_checker.models import (
    ActDocument,
    CheckRequest,
    ComplianceResult,
    EliRef,
    LegalUnit,
    LlmEvaluationPayload,
)


class GraphState(TypedDict, total=False):
    request: CheckRequest
    eli_ref: EliRef
    act: ActDocument  # metadata (title, source_url) + cleaned text
    units: list[LegalUnit]
    candidates: list[tuple[LegalUnit, float]]  # BM25 results (unit, score)
    messages: Annotated[list[AnyMessage], add_messages]  # research conversation (AI + ToolMessage)
    llm_payload: LlmEvaluationPayload | None
    schema_errors: list[str]
    eval_attempts: int
    result: ComplianceResult
