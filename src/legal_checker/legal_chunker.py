"""Splitting an act's text along its legal structure (SPEC Task 3.1)."""

import re

from pydantic import BaseModel

# Parent headings, outermost first. A heading is a line of its own: the level keyword and
# a designator ("Rozdział 5a", "TYTUŁ VI^1", "KSIĘGA PIERWSZA"), followed by the heading's
# title on the next line(s). The keyword must be capitalised, so a wrapped line of body text
# ("rozdział 2") is not taken for one.
_LEVELS = ["Księga", "Tytuł", "Dział", "Rozdział", "Oddział"]
_HEADING = re.compile(
    r"^(" + "|".join(f"{level}|{level.upper()}" for level in _LEVELS) + r")\s+((?:\d+[a-z]*|[IVXLCDM]+[a-z]*)(?:\^\d+)?|[A-ZĄĆĘŁŃÓŚŹŻ]+)$"
)
_ROMAN = re.compile(r"[IVXLCDM]+")
# Start of an article or of a "§" unit (regulations); ends a heading's title.
_UNIT_START = re.compile(r"^(?:Art\.|§)\s*\d+[a-z]*")
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
