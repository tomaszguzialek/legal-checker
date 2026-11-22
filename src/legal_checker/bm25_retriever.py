"""In-memory BM25 retrieval over an act's units (SPEC Task 3.2)."""

import re

from rank_bm25 import BM25Okapi

from legal_checker.models import LegalUnit

# Words and numbers; punctuation separates tokens ("art. 27a," -> "art", "27a").
_WORD = re.compile(r"\w+")
# Frequent function words that carry no meaning for retrieval.
STOPWORDS = frozenset(
    "a albo ani aby by być czy dla do i ich jak jako jest jego jej już lub na nad o od oraz po pod"
    " przez przy się są ta tak te tego to tym u w we z za ze że który która które których którym".split()
)
# Inflectional and common derivational endings, longest first so that "ami" wins over "i".
# This is a light suffix stripper, not a full Polish stemmer: it maps the forms that matter
# for retrieval ("odstąpienie", "odstąpienia", "odstąpić", "odstąpił") to one stem ("odstąp").
_SUFFIXES = sorted(
    """
    ieniem aniem eniem owania owanie owaniu
    ienia ienie ieniu ania anie aniu enia enie eniu ości ość ować owej owym owych owie owi ował
    ach ami ego emu ich imi iej ymi ych ową owa owe owy ały ało ała ali ili ują uje
    om ów ie ej ym im ią ię em ał ił ić ać eć ą ę
    a e i o u y
    """.split(),
    key=len,
    reverse=True,
)
# A reference to an article of the same act: "o którym mowa w art. 27", "art. 43^1 ust. 2 pkt 1".
# References to other acts ("art. 385^1 Kodeksu cywilnego", "art. 6 ust. 1 rozporządzenia 2016/679")
# are skipped: the act's name follows the number and its paragraph, point and letter.
# Groups: article number, paragraph number (if given).
_NUMBER = r"\d+[a-z]*(?:\^\d+[a-z]*)?"
_REFERENCE = re.compile(
    rf"\bart\.\s*({_NUMBER})(?![\w^])(?:\s+ust\.\s*({_NUMBER})(?![\w^]))?"
    r"(?!(?:\s+(?:ust\.|pkt|lit\.)\s*[\w^]+)*\s+(?:ustaw|kodeks|rozporządze|dyrektyw|k\.\s?[a-z]\.))",
    re.IGNORECASE,
)
# Stems shorter than this are not trimmed further: "dni" stays rather than becoming "dn".
_MIN_STEM = 3


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens of Polish text, without stopwords, each reduced to its stem."""
    return [stem(word) for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


def stem(word: str) -> str:
    """Strip the longest known ending that leaves at least ``_MIN_STEM`` characters; numbers stay.

    "ó" becomes "o", as it alternates within a word's forms ("umowa" / "umów", "sposób" / "sposobu").
    """
    if not word.isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            word = word[: -len(suffix)]
            break
    return word.replace("ó", "o")


class InMemoryBm25Index:
    """BM25 index over an act's units, held in RAM for the duration of one check.

    A unit is indexed with its parent headings, so a chapter title ("Prawo odstąpienia od umowy")
    also matches the articles under it.
    """

    def __init__(self, units: list[LegalUnit]):
        if not units:
            raise ValueError("Cannot index an act without units")
        self.units = units
        self.corpus = [tokenize(" ".join([*unit.hierarchy_path, unit.text])) for unit in units]
        self.bm25 = BM25Okapi(self.corpus)
        self._by_article: dict[str, list[int]] = {}
        for i, unit in enumerate(units):
            self._by_article.setdefault(unit.article, []).append(i)

    def search(self, query: str, top_k: int = 12) -> list[tuple[LegalUnit, float]]:
        """The ``top_k`` best-matching units, best first, plus the articles they refer to.

        Units sharing no token with the query are left out. Units of articles referenced by
        the top results ("o którym mowa w art. 27") follow them with their own scores, even
        if that makes the list longer than ``top_k``. Of an article split into paragraphs, only
        the referenced paragraph is added ("art. 39 ust. 3"), or all of them if none is named.
        """
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: -scores[i])
        found = ranked[:top_k]
        referenced: list[int] = []
        for article, paragraph in self._references([self.units[i] for i in found]):
            for i in self._by_article.get(article, []):
                if paragraph is None or self.units[i].paragraph in (None, paragraph):
                    if i not in found and i not in referenced:
                        referenced.append(i)
        return [(self.units[i], float(scores[i])) for i in found + referenced]

    def get(self, article: str, paragraph: str | None = None) -> LegalUnit | None:
        """The unit of an article (or of its paragraph), looked up by number, e.g. ``get("27a", "2")``.

        An article split into paragraphs is returned whole when no paragraph is asked for; an
        article kept whole is returned for any of its paragraphs. ``None`` if there is no such unit.
        """
        units = [self.units[i] for i in self._by_article.get(_normalize_number(article), [])]
        if not units:
            return None
        if paragraph is not None:
            paragraph = _normalize_number(paragraph)
            return next((unit for unit in units if unit.paragraph in (paragraph, None)), None)
        if len(units) == 1:
            return units[0]
        first = units[0]
        return first.model_copy(
            update={
                "unit_id": first.unit_id.rsplit(".", 2)[0] if first.paragraph else first.unit_id,
                "paragraph": None,
                "hierarchy_path": first.hierarchy_path[:-1],
                "text": "\n".join(unit.text for unit in units),
            }
        )

    @staticmethod
    def _references(units: list[LegalUnit]) -> list[tuple[str, str | None]]:
        """(article, paragraph) the units refer to, in order of appearance, without their own article."""
        references: list[tuple[str, str | None]] = []
        for unit in units:
            for match in _REFERENCE.finditer(unit.text):
                if match[1] != unit.article and (reference := (match[1], match[2])) not in references:
                    references.append(reference)
        return references


def _normalize_number(number: str) -> str:
    """Unit number as the chunker writes it: "Art. 27 a" -> "27a"."""
    return re.sub(r"^(?:art\.|ust\.|§)|\s+", "", number.strip(), flags=re.IGNORECASE)


class LateBoundBm25Index:
    """A request's index, built once the act's units are known.

    The agent tools and the research model are created before the graph runs, but the act's
    units only come out of its ``chunk_act`` node: the ``retrieve`` node builds the index here,
    and the tools look it up when called.
    """

    def __init__(self) -> None:
        self._index: InMemoryBm25Index | None = None

    def build(self, units: list[LegalUnit]) -> InMemoryBm25Index:
        self._index = InMemoryBm25Index(units)
        return self._index

    @property
    def index(self) -> InMemoryBm25Index:
        if self._index is None:
            raise RuntimeError("The BM25 index is not built yet: the graph has not reached its retrieve node")
        return self._index

    def search(self, query: str, top_k: int = 12) -> list[tuple[LegalUnit, float]]:
        return self.index.search(query, top_k=top_k)

    def get(self, article: str, paragraph: str | None = None) -> LegalUnit | None:
        return self.index.get(article, paragraph)
