"""The compliance-check StateGraph and its entry point (SPEC Task 4.3, topology in section 3.5)."""

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from legal_checker.agent.llm import build_models
from legal_checker.agent.nodes import (
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
from legal_checker.agent.state import GraphState
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import LateBoundBm25Index
from legal_checker.config import Settings, get_settings
from legal_checker.eli_client import EliClient
from legal_checker.models import CheckRequest, ComplianceResult


def build_graph(
    research_model: Runnable,
    eval_model: Runnable,
    tools: list[BaseTool],
    *,
    eli: EliClient,
    index: LateBoundBm25Index,
    settings: Settings,
) -> CompiledStateGraph:
    """Compile the graph for one request.

    ``tools`` must search ``index``, which the ``retrieve`` node builds from the fetched act.
    Both models are injected, so tests run the graph on fake ones.
    """
    graph = StateGraph(GraphState)
    for name, node in make_act_nodes(eli, index, settings.bm25_top_k).items():
        graph.add_node(name, node)
    graph.add_node("short_circuit_na", short_circuit_na)
    graph.add_node("research", make_research_node(research_model, settings.agent_max_tool_calls))
    graph.add_node("tools", make_tools_node(tools, settings.agent_max_tool_calls))
    graph.add_node("evaluate", make_evaluate_node(eval_model))
    graph.add_node("validate", validate)

    graph.add_edge(START, "resolve_act")
    graph.add_edge("resolve_act", "fetch_act")
    graph.add_edge("fetch_act", "chunk_act")
    graph.add_edge("chunk_act", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["short_circuit_na", "research"])
    graph.add_edge("short_circuit_na", END)
    graph.add_conditional_edges(
        "research", make_route_after_research(settings.agent_max_tool_calls), ["tools", "evaluate"]
    )
    graph.add_edge("tools", "research")
    graph.add_conditional_edges("evaluate", route_after_evaluate, ["evaluate", "validate"])
    graph.add_edge("validate", END)
    return graph.compile()


def check(request: CheckRequest, settings: Settings | None = None) -> ComplianceResult:
    """Check a document fragment against the act its hint points to.

    Creates the request's index, tools and models, then runs the graph. Raises ``ActResolutionError``,
    ``EliClientError`` or ``EvaluationError`` when the act cannot be found or fetched, or the
    model keeps violating the output schema.
    """
    settings = settings or get_settings()
    index = LateBoundBm25Index()
    tools = make_tools(index)
    research_model, eval_model = build_models(settings, tools)
    with EliClient.from_settings(settings) as eli:
        graph = build_graph(research_model, eval_model, tools, eli=eli, index=index, settings=settings)
        state = graph.invoke({"request": request}, config={"recursion_limit": settings.graph_recursion_limit})
    return state["result"]
