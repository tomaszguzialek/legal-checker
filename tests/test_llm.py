import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from legal_checker.agent import llm
from legal_checker.agent.llm import build_models
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import InMemoryBm25Index
from legal_checker.config import Settings
from legal_checker.models import LegalUnit, LlmEvaluationPayload

TOOLS = make_tools(
    InMemoryBm25Index(
        [
            LegalUnit(
                unit_id="DU/2014/827:art.27",
                eli="DU/2014/827",
                article="27",
                hierarchy_path=["Rozdział 4", "Art. 27"],
                text="Art. 27. Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić.",
            )
        ]
    )
)


class RecordingChatModel:
    """Stands in for the provider model: records how build_models configures it."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.bound_tools = None
        self.structured_output = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return RunnableLambda(lambda _: AIMessage(content=""))

    def with_structured_output(self, schema, **kwargs):
        self.structured_output = (schema, kwargs)
        return RunnableLambda(lambda _: {"raw": None, "parsed": None, "parsing_error": None})


@pytest.fixture
def created(monkeypatch) -> list[RecordingChatModel]:
    models: list[RecordingChatModel] = []

    def init_chat_model(model: str, **kwargs) -> RecordingChatModel:
        models.append(RecordingChatModel(model, **kwargs))
        return models[-1]

    monkeypatch.setattr(llm, "init_chat_model", init_chat_model)
    return models


def test_both_models_come_from_one_deterministic_chat_model(created):
    build_models(Settings(llm_model="anthropic:claude-sonnet-5-5"), TOOLS)

    assert len(created) == 1
    assert created[0].model == "anthropic:claude-sonnet-5-5"
    assert created[0].kwargs == {"temperature": 0.0}


def test_research_model_has_the_request_tools_bound(created):
    research_model, _ = build_models(Settings(), TOOLS)

    assert created[0].bound_tools == TOOLS
    assert isinstance(research_model.invoke("pytanie"), AIMessage)


def test_eval_model_returns_the_payload_schema_with_raw_output(created):
    _, eval_model = build_models(Settings(), TOOLS)

    assert created[0].structured_output == (LlmEvaluationPayload, {"method": "json_schema", "include_raw": True})
    assert set(eval_model.invoke("pytanie")) == {"raw", "parsed", "parsing_error"}


@pytest.mark.parametrize("llm_model", ["anthropic:claude-sonnet-5-5", "openai:gpt-5", "google_genai:gemini-3-pro"])
def test_provider_is_chosen_only_by_the_llm_model_setting(created, llm_model):
    build_models(Settings(llm_model=llm_model), TOOLS)

    assert created[0].model == llm_model


@pytest.mark.live
def test_configured_provider_calls_tools_and_returns_the_payload():
    research_model, eval_model = build_models(Settings.from_env(), TOOLS)

    research = research_model.invoke(
        [HumanMessage("Pobierz narzędziem get_provision treść art. 27 tego aktu. Nie odpowiadaj tekstem.")]
    )
    evaluation = eval_model.invoke(
        [
            HumanMessage(
                "Przepis: Art. 27. Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić.\n"
                "Fragment dokumentu: Klient może odstąpić od umowy w terminie 7 dni.\n"
                "Oceń zgodność fragmentu z przepisem."
            )
        ]
    )

    assert [call["name"] for call in research.tool_calls] == ["get_provision"]
    assert research.tool_calls[0]["args"]["article"] == "27"
    assert evaluation["parsing_error"] is None
    assert isinstance(evaluation["parsed"], LlmEvaluationPayload)
