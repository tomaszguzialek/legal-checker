from legal_checker.agent.prompts import EVALUATION_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import InMemoryBm25Index
from legal_checker.models import ComplianceStatus, LegalUnit, LlmEvaluationPayload

TOOL_NAMES = [
    tool.name
    for tool in make_tools(
        InMemoryBm25Index(
            [LegalUnit(unit_id="DU/2014/827:art.1", eli="DU/2014/827", article="1", hierarchy_path=["Art. 1"], text="Art. 1.")]
        )
    )
]


def test_research_prompt_names_every_agent_tool():
    assert all(name in RESEARCH_SYSTEM_PROMPT for name in TOOL_NAMES)


def test_research_prompt_refers_to_the_tagged_sections_of_the_user_message():
    assert "<document_fragment>" in RESEARCH_SYSTEM_PROMPT
    assert "<legal_provisions>" in RESEARCH_SYSTEM_PROMPT


def test_research_prompt_forbids_speculative_searches_and_allows_stopping_without_tools():
    assert "Nie szukaj „na zapas”" in RESEARCH_SYSTEM_PROMPT
    assert "zakończ bez wywołania narzędzi" in RESEARCH_SYSTEM_PROMPT


def test_evaluation_prompt_refers_to_the_tagged_sections_of_the_user_message():
    assert all(tag in EVALUATION_SYSTEM_PROMPT for tag in ["<document_fragment>", "<legal_provisions>", "<tool_results>"])


def test_evaluation_prompt_names_every_status_exactly_as_the_schema_does():
    assert all(f"„{status.value}”" in EVALUATION_SYSTEM_PROMPT for status in ComplianceStatus)


def test_evaluation_prompt_names_only_fields_of_the_payload():
    fields = [
        "document_excerpt",
        "rationale",
        "matched_provision",
        "applicability_unclear",
        "confidence",
        "suggested_fix",
    ]

    assert set(fields) <= set(LlmEvaluationPayload.model_fields)
    assert all(field in EVALUATION_SYSTEM_PROMPT for field in fields)


def test_evaluation_prompt_states_the_hard_rules():
    assert "Oceniaj wyłącznie na podstawie przepisów z sekcji <legal_provisions> oraz przepisów zwróconych przez narzędzia." in EVALUATION_SYSTEM_PROMPT
    assert "Nie dopowiadaj przepisów spoza kontekstu" in EVALUATION_SYSTEM_PROMPT
    assert "dosłowny wycinek tekstu z <document_fragment>" in EVALUATION_SYSTEM_PROMPT
    assert "2–5 zwięzłych zdań" in EVALUATION_SYSTEM_PROMPT
    assert "czy akt w ogóle ma zastosowanie do stron" in EVALUATION_SYSTEM_PROMPT
    assert "confidence nie wyższe niż 0.7" in EVALUATION_SYSTEM_PROMPT
