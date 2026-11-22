from pathlib import Path

from langchain_core.messages import ToolCall, ToolMessage

from legal_checker.agent.tools import MAX_CONTENT_CHARS, make_tools
from legal_checker.bm25_retriever import InMemoryBm25Index
from legal_checker.legal_chunker import chunk_act
from legal_checker.models import LegalUnit


def unit(article: str, text: str) -> LegalUnit:
    return LegalUnit(
        unit_id=f"DU/2014/827:art.{article}",
        eli="DU/2014/827",
        article=article,
        hierarchy_path=["Rozdział 4", f"Art. {article}"],
        text=text,
    )


UNITS = [
    unit("27", "Art. 27. Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić."),
    unit("32", "Art. 32. Przedsiębiorca ma obowiązek niezwłocznie zwrócić konsumentowi wszystkie dokonane płatności."),
    unit("44", "Art. 44. Przepisy rozdziału stosuje się do umów o dostarczanie treści cyfrowych."),
]
# With few units, BM25 weighs terms found in half of them or more as zero; these keep the corpus realistic.
UNRELATED = [unit(str(article), f"Art. {article}. Minister właściwy określi wzór formularza nr {article}.") for article in range(90, 95)]


def tools_by_name(units: list[LegalUnit] = UNITS) -> dict:
    return {tool.name: tool for tool in make_tools(InMemoryBm25Index(units))}


def test_make_tools_builds_search_and_get_with_polish_descriptions():
    tools = tools_by_name()

    assert set(tools) == {"search_provisions", "get_provision"}
    assert tools["search_provisions"].description.startswith("Wyszukuje w treści aktu")
    assert tools["get_provision"].description.startswith("Pobiera dokładną treść przepisu")
    assert all(tool.response_format == "content_and_artifact" for tool in tools.values())


def test_tool_argument_schemas_are_typed_and_described():
    tools = tools_by_name()

    search = tools["search_provisions"].args_schema.model_json_schema()
    get = tools["get_provision"].args_schema.model_json_schema()

    assert search["required"] == ["query"]
    assert search["properties"]["top_k"]["type"] == "integer"
    assert search["properties"]["top_k"]["default"] == 5
    assert get["required"] == ["article"]
    assert get["properties"]["paragraph"]["default"] is None
    assert all(prop["description"] for prop in [*search["properties"].values(), *get["properties"].values()])


def test_tools_are_bound_to_their_own_index():
    first = tools_by_name()
    second = tools_by_name([unit("1", "Art. 1. Ustawa określa prawa konsumenta.")])

    assert first["get_provision"].func(article="27")[1] == ["DU/2014/827:art.27"]
    assert second["get_provision"].func(article="27")[1] == []


def test_content_heads_each_provision_with_its_unit_id_and_hierarchy_path():
    content, artifact = tools_by_name()["search_provisions"].func(query="dokonane płatności")

    assert artifact == ["DU/2014/827:art.32"]
    assert content == f"[DU/2014/827:art.32] Rozdział 4 > Art. 32\n{UNITS[1].text}"


def test_no_results_give_a_readable_message_and_an_empty_artifact():
    tools = tools_by_name()

    assert tools["search_provisions"].func(query="godziny spotkań zespołu") == (
        "Brak przepisów pasujących do zapytania „godziny spotkań zespołu”. Spróbuj innych słów kluczowych.",
        [],
    )
    assert tools["get_provision"].func(article="99", paragraph="2") == ("Nie znaleziono przepisu art. 99 ust. 2 w tym akcie.", [])


def test_content_is_cut_to_the_limit_and_lists_only_the_units_shown():
    text = "Konsument może odstąpić od umowy. " * 120  # ~4k characters per unit
    units = [unit(str(article), f"Art. {article}. {text}") for article in range(1, 5)] + UNRELATED

    content, artifact = tools_by_name(units)["search_provisions"].func(query="odstąpić od umowy")

    assert artifact == ["DU/2014/827:art.1", "DU/2014/827:art.2", "DU/2014/827:art.3"]
    assert content.count(" […]") == 1
    assert content.endswith("[Pominięte z powodu limitu długości odpowiedzi: DU/2014/827:art.4]")
    assert len(content) <= MAX_CONTENT_CHARS + 100


# Full text of the consumer rights act (DU/2014/827), as in test_bm25_retriever.py.
FULL_ACT_TOOLS = {
    tool.name: tool
    for tool in make_tools(
        InMemoryBm25Index(
            chunk_act("DU/2014/827", (Path(__file__).parent / "fixtures" / "acts" / "DU_2014_827.txt").read_text()).units
        )
    )
}


def call(name: str, **args) -> ToolMessage:
    """Invoke a tool the way LangGraph's ToolNode does: with the model's ToolCall."""
    return FULL_ACT_TOOLS[name].invoke(ToolCall(name=name, args=args, id=f"call_{name}", type="tool_call"))


def test_search_provisions_invoked_with_a_tool_call_returns_a_tool_message_with_unit_ids():
    message = call("search_provisions", query="odstąpienie od umowy zawartej na odległość w terminie 14 dni", top_k=3)

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_search_provisions"
    assert message.name == "search_provisions"
    assert "DU/2014/827:art.27" in message.artifact
    assert message.content.startswith(f"[{message.artifact[0]}] Rozdział 4 Prawo odstąpienia od umowy")
    assert all(f"[{unit_id}] " in message.content for unit_id in message.artifact)
    assert len(message.content) <= MAX_CONTENT_CHARS


def test_get_provision_invoked_with_a_tool_call_returns_the_article():
    message = call("get_provision", article="27")

    assert message.artifact == ["DU/2014/827:art.27"]
    assert message.content.startswith("[DU/2014/827:art.27] Rozdział 4 Prawo odstąpienia")
    assert "Art. 27. 1. Konsument, który zawarł umowę na odległość" in message.content
    assert "14 dni" in message.content


def test_get_provision_returns_only_the_requested_paragraph_of_a_split_article():
    message = call("get_provision", article="38", paragraph="1")

    assert message.artifact == ["DU/2014/827:art.38.ust.1"]
    assert "> Art. 38 > ust. 1\n" in message.content


def test_get_provision_of_a_missing_article_returns_a_message_and_no_units():
    message = call("get_provision", article="999")

    assert message.content == "Nie znaleziono przepisu art. 999 w tym akcie."
    assert message.artifact == []
    assert message.status == "success"


def test_search_provisions_with_no_matching_words_returns_a_message_and_no_units():
    message = call("search_provisions", query="xyzzy qwerty")

    assert message.content.startswith("Brak przepisów pasujących do zapytania „xyzzy qwerty”")
    assert message.artifact == []
