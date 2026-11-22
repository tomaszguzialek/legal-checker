"""Read-only agent tools over the current request's BM25 index (SPEC Task 4.1)."""

from langchain_core.tools import BaseTool, tool

from legal_checker.bm25_retriever import InMemoryBm25Index, LateBoundBm25Index
from legal_checker.models import LegalUnit

# Upper bound on one tool result's text (~2.5k tokens): a few full articles, not half the act.
MAX_CONTENT_CHARS = 10_000
# A unit that would overflow the limit is cut, unless so little room is left that only its header would fit.
_MIN_CUT_BLOCK_CHARS = 200
_CUT_MARK = " […]"
_SEPARATOR = "\n\n"


def make_tools(index: InMemoryBm25Index | LateBoundBm25Index) -> list[BaseTool]:
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
        return _result(
            [unit for unit, _ in index.search(query, top_k=top_k)],
            f"Brak przepisów pasujących do zapytania „{query}”. Spróbuj innych słów kluczowych.",
        )

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
        provision = f"art. {article}" + (f" ust. {paragraph}" if paragraph else "")
        return _result([unit] if unit else [], f"Nie znaleziono przepisu {provision} w tym akcie.")

    return [search_provisions, get_provision]


def _result(units: list[LegalUnit], empty_message: str) -> tuple[str, list[str]]:
    """Provisions as ``[unit_id] hierarchy_path`` blocks, cut to ``MAX_CONTENT_CHARS``.

    The artifact lists only units the LLM actually got to see, in part or whole.
    """
    if not units:
        return empty_message, []
    blocks: list[str] = []
    shown: list[str] = []
    remaining = MAX_CONTENT_CHARS
    for unit in units:
        block = f"[{unit.unit_id}] {' > '.join(unit.hierarchy_path)}\n{unit.text}"
        if len(block) > remaining:
            if remaining >= _MIN_CUT_BLOCK_CHARS:
                blocks.append(block[:remaining].rstrip() + _CUT_MARK)
                shown.append(unit.unit_id)
            break
        blocks.append(block)
        shown.append(unit.unit_id)
        remaining -= len(block) + len(_SEPARATOR)
    if len(shown) < len(units):
        skipped = ", ".join(unit.unit_id for unit in units[len(shown) :])
        blocks.append(f"[Pominięte z powodu limitu długości odpowiedzi: {skipped}]")
    return _SEPARATOR.join(blocks), shown
