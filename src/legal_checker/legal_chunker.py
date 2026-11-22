"""Splitting an act's text along its legal structure (SPEC Task 3.1)."""

import re

from pydantic import BaseModel

from legal_checker.models import LegalUnit

# Parent headings, outermost first. A heading is a line of its own: the level keyword and
# a designator ("Rozdział 5a", "TYTUŁ VI^1", "KSIĘGA PIERWSZA"), followed by the heading's
# title on the next line(s). The keyword must be capitalised, so a wrapped line of body text
# ("rozdział 2") is not taken for one.
_LEVELS = ["Księga", "Tytuł", "Dział", "Rozdział", "Oddział"]
_HEADING = re.compile(
    r"^(" + "|".join(f"{level}|{level.upper()}" for level in _LEVELS) + r")\s+"
    r"((?:\d+[a-z]*|[IVXLCDM]+[a-z]*)(?:\^\d+)?|[A-ZĄĆĘŁŃÓŚŹŻ]+)$"
)
_ROMAN = re.compile(r"[IVXLCDM]+")
# Start of an article or of a "§" unit (regulations); ends a heading's title.
_UNIT_START = re.compile(r"^(?:Art\.|§)\s*\d+[a-z]*")
# Unit boundaries. Articles may be inserted ("43a", "449^1") or listed as a range when
# repealed ("Art. 44–50. (pominięte)"). "§" starts a unit only in acts without articles
# (regulations); in codes it numbers the paragraphs of an article ("Art. 8. § 1.").
_NUMBER = r"\d+[a-z]*(?:\^\d+[a-z]*)?"
_ARTICLE = re.compile(rf"^Art\.\s*({_NUMBER}(?:\s*[–-]\s*{_NUMBER})?)\.")
_PARAGRAPH_UNIT = re.compile(rf"^§\s*({_NUMBER})\.")
# Units are kept whole up to this length; longer ones are split into their paragraphs
# ("ust." in statutes and regulations, "§" in codes), each prefixed with its unit number.
MAX_UNIT_CHARS = 3000
_SUBPARAGRAPH = re.compile(rf"^({_NUMBER})\.\s")
_CODE_PARAGRAPH = re.compile(rf"^§\s*({_NUMBER})\.\s")
# Lines starting a structural element: a unit, a paragraph ("2.", "§ 2."), a point ("3)")
# or a letter ("b)"). Any other line is a PDF line wrapped mid-sentence. Indents ("–") are
# not detected: a wrapped line of a definition ("umowa … \n– umowę …") starts the same way.
_ELEMENT_START = re.compile(rf"^(?:Art\.\s*\d|§\s*\d|{_NUMBER}[.)]\s|[a-z]{{1,2}}\)\s)")
# A range wrapped after its dash ("pkt 10–" / "20") is rejoined without a space.
_OPEN_RANGE = re.compile(r"\d[–-]$")
# Titles span 1-2 lines in practice; more lines after a heading are body text of an untitled one.
_MAX_TITLE_LINES = 3


class Section(BaseModel):
    """Consecutive body lines under the same parent headings."""

    hierarchy_path: list[str]  # e.g. ["Księga pierwsza Część ogólna", "Tytuł I Przepisy wstępne"]
    lines: list[str]


def split_sections(text: str) -> list[Section]:
    """Walk the text line by line, keeping a stack of the enclosing parent headings.

    A heading closes every open heading of its own or a lower level. Text before the first
    heading (the act's title) gets an empty path.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level index, label with title)
    lines: list[str] = []
    title: list[str] | None = None  # title lines of the heading being read, if any

    def close_heading() -> None:
        nonlocal title
        if title is not None:
            level, label = stack.pop()
            stack.append((level, " ".join([label, *title])))
            title = None

    def flush() -> None:
        if lines:
            sections.append(Section(hierarchy_path=[label for _, label in stack], lines=lines.copy()))
            lines.clear()

    for line in text.split("\n"):
        if heading := _HEADING.match(line):
            close_heading()
            flush()
            level = _LEVELS.index(heading[1].capitalize())
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, f"{_LEVELS[level]} {_designator(heading[2])}"))
            title = []
        elif title is not None and not _UNIT_START.match(line) and len(title) < _MAX_TITLE_LINES:
            title.append(line)
        else:
            close_heading()
            lines.append(line)
    close_heading()
    flush()
    return sections


def _designator(designator: str) -> str:
    """Numbers and Roman numerals as they are, ordinal words in lower case ("PIERWSZA" -> "pierwsza")."""
    if designator.isalpha() and not _ROMAN.fullmatch(designator):
        return designator.lower()
    return designator


def chunk_act(eli: str, text: str, max_unit_chars: int = MAX_UNIT_CHARS) -> list[LegalUnit]:
    """Split an act into its articles (or "§" units of a regulation), each with its parent headings.

    Each paragraph, point and letter of a unit starts on a new line. Units longer than
    ``max_unit_chars`` are split into their paragraphs, if they have any. Text before the
    first unit of a section (the act's title, preamble) is not part of any unit.
    """
    sections = split_sections(text)
    uses_articles = any(_ARTICLE.match(line) for section in sections for line in section.lines)
    boundary, label, id_prefix = (_ARTICLE, "Art.", "art.") if uses_articles else (_PARAGRAPH_UNIT, "§", "par.")

    units: list[LegalUnit] = []
    for hierarchy_path, start, lines in _unit_lines(sections, boundary):
        number = re.sub(r"\s+", "", start[1])
        unit = LegalUnit(
            unit_id=f"{eli}:{id_prefix}{number}",
            eli=eli,
            article=number,
            hierarchy_path=[*hierarchy_path, f"{label} {number}"],
            text="\n".join(lines),
        )
        paragraphs = _split_paragraphs(lines, start.end(), code_paragraphs=uses_articles)
        if len(unit.text) <= max_unit_chars or not paragraphs:
            units.append(unit)
            continue
        for paragraph_label, paragraph_id, paragraph, paragraph_lines in paragraphs:
            units.append(
                unit.model_copy(
                    update={
                        "unit_id": f"{unit.unit_id}.{paragraph_id}",
                        "paragraph": paragraph,
                        "hierarchy_path": [*unit.hierarchy_path, paragraph_label],
                        "text": "\n".join(paragraph_lines),
                    }
                )
            )
    return units


def _unit_lines(sections: list[Section], boundary: re.Pattern) -> list[tuple[list[str], re.Match, list[str]]]:
    """Group each section's lines into units: (parent headings, unit start match, element lines)."""
    units: list[tuple[list[str], re.Match, list[str]]] = []
    for section in sections:
        lines: list[str] | None = None  # lines of the current unit; none before the section's first unit
        for line in section.lines:
            if start := boundary.match(line):
                lines = [line]
                units.append((section.hierarchy_path, start, lines))
            elif lines is None:
                continue
            elif _ELEMENT_START.match(line):
                lines.append(line)
            else:
                lines[-1] += ("" if _OPEN_RANGE.search(lines[-1]) and line[0].isdigit() else " ") + line
    return units


def _split_paragraphs(
    lines: list[str], number_end: int, code_paragraphs: bool
) -> list[tuple[str, str, str, list[str]]]:
    """Split a unit's element lines into paragraphs: (label, id suffix, number, lines).

    The first paragraph follows the unit number on its first line ("Art. 12. 1. …", "Art. 26.
    § 1. …") and must be number 1; without it the unit has no paragraphs and ``[]`` is returned.
    Later paragraphs get the unit number as a prefix ("Art. 12. 2. …") to keep their context.
    """
    prefix, first = lines[0][:number_end], lines[0][number_end:].lstrip()
    for pattern, label, id_prefix in [(_SUBPARAGRAPH, "ust.", "ust."), (_CODE_PARAGRAPH, "§", "par.")]:
        if pattern is _CODE_PARAGRAPH and not code_paragraphs:
            continue
        if (match := pattern.match(first)) and match[1] == "1":
            break
    else:
        return []

    paragraphs: list[tuple[str, str, str, list[str]]] = []
    for i, line in enumerate(lines):
        if i == 0 or (match := pattern.match(line)):
            number = "1" if i == 0 else match[1]
            text = line if i == 0 else f"{prefix} {line}"
            paragraphs.append((f"{label} {number}", f"{id_prefix}{number}", number, [text]))
        else:
            paragraphs[-1][3].append(line)
    return paragraphs
