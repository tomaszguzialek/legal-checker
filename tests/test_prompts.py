from legal_checker.agent.prompts import RESEARCH_SYSTEM_PROMPT
from legal_checker.agent.tools import make_tools
from legal_checker.bm25_retriever import InMemoryBm25Index
from legal_checker.models import LegalUnit

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
