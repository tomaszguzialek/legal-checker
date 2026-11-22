"""Read-only agent tools over the current request's BM25 index (SPEC Task 4.1)."""

from langchain_core.tools import BaseTool, tool

from legal_checker.bm25_retriever import InMemoryBm25Index
from legal_checker.models import LegalUnit


def make_tools(index: InMemoryBm25Index) -> list[BaseTool]:
    """Tools bound to one request's index, so no state is shared between checks.

    Each returns ``(content, artifact)``: the provisions' text for the LLM and their
    ``unit_id``s, from which the post-validator derives what the agent has seen.
    """

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_provisions(query: str, top_k: int = 5) -> tuple[str, list[str]]:
        """Wyszukuje w treści aktu dodatkowe przepisy pasujące do zapytania (BM25).

        Użyj, gdy fragment dokumentu dotyczy pojęcia lub obowiązku, którego nie obejmują
        przepisy z kontekstu, np. "termin zwrotu płatności" albo "koszty odesłania rzeczy".

        Args:
            query: Słowa kluczowe po polsku, najlepiej sformułowania z treści ustawy.
            top_k: Maksymalna liczba zwróconych przepisów.
        """
        return _result([unit for unit, _ in index.search(query, top_k=top_k)])

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def get_provision(article: str, paragraph: str | None = None) -> tuple[str, list[str]]:
        """Pobiera dokładną treść przepisu aktu po numerze artykułu (i opcjonalnie ustępu).

        Użyj, gdy przepis z kontekstu odsyła do innego artykułu tego samego aktu,
        np. "o którym mowa w art. 27", a jego treści nie ma w kontekście.

        Args:
            article: Numer artykułu, np. "27" lub "43a".
            paragraph: Numer ustępu, np. "2"; pomiń, aby pobrać cały artykuł.
        """
        unit = index.get(article, paragraph)
        return _result([unit] if unit else [])

    return [search_provisions, get_provision]


def _result(units: list[LegalUnit]) -> tuple[str, list[str]]:
    return "\n\n".join(unit.text for unit in units), [unit.unit_id for unit in units]
