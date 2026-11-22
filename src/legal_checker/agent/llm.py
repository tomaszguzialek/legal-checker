"""Chat model factory for the research and evaluation steps (SPEC Task 4.2)."""

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from legal_checker.config import Settings
from legal_checker.models import LlmEvaluationPayload


def build_models(settings: Settings, tools: list[BaseTool]) -> tuple[Runnable, Runnable]:
    """Research model with the request's tools bound, and the structured-output evaluation model.

    The provider comes only from ``settings.llm_model`` (``"provider:model"``), so switching
    e.g. from ``anthropic:...`` to ``openai:...`` is a ``.env`` change; its integration
    package (``legal-checker[openai]``) must be installed. The graph takes both models as
    arguments, so tests pass fake ones instead of calling this.

    The evaluation model returns ``{"raw", "parsed", "parsing_error"}`` (``include_raw=True``),
    so a schema violation reaches the graph as data to retry on, not as an exception.
    """
    model = init_chat_model(settings.llm_model, temperature=0.0)
    research_model = model.bind_tools(tools)
    eval_model = model.with_structured_output(LlmEvaluationPayload, method="json_schema", include_raw=True)
    return research_model, eval_model
