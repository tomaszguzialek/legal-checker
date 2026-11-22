from legal_checker.agent.tools import make_tools
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
]


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

    assert first["get_provision"].func(article="27") == (UNITS[0].text, ["DU/2014/827:art.27"])
    assert second["get_provision"].func(article="27") == ("", [])
