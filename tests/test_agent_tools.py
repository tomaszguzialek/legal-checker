from legal_checker.agent.tools import MAX_CONTENT_CHARS, make_tools
from legal_checker.bm25_retriever import InMemoryBm25Index
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
