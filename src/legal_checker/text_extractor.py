"""Text extraction from act PDFs and HTML (SPEC Task 2.3)."""

import re
from collections import Counter

import pymupdf
from bs4 import BeautifulSoup, NavigableString
from pydantic import BaseModel, Field

from legal_checker.eli_client import ActContent

# A span is body text if its font is at least the body size minus this. Smaller spans are
# footnotes, margin notes, running headers/footers and superscript footnote markers. The
# body size is the document's dominant size, except on pages without any text of that size:
# annexes are set entirely in a smaller font, so there the page's own dominant size is used.
_SIZE_TOLERANCE = 0.5
# Spans whose vertical centres differ by less than this (pt) share a visual line, e.g.
# a point number "1)" and its text, which the PDF stores as separate lines.
_LINE_TOLERANCE = 3.0
# A horizontal gap wider than this (pt) between adjacent spans is a word break.
_WORD_GAP = 1.0

_WHITESPACE = re.compile(r"\s+")

# Sanity checks: the size filter fails silently when the layout changes, so suspicious
# results are reported as warnings instead (they end up in ComplianceResult.warnings).
_MAX_DROPPED_SHARE = 0.25
_UNIT_START = re.compile(r"^(?:Art\.|§)\s*\d+[a-z]*\.")
_UNIT_LINE = re.compile(r"^(?:Art\.|§)\s*\d+", re.MULTILINE)

# Running headers/footers repeat on most pages (page numbers aside) as a page's first or
# last line. A line starting like a provision ("Art. 5. (uchylony)") is never one.
_EDGE_LINES = 1
_MIN_REPEAT_PAGES = 3
_MIN_REPEAT_SHARE = 0.5
_DIGITS = re.compile(r"\d+")
# Known page furniture, as a fallback for documents too short for repetition detection.
_FURNITURE = re.compile(
    r"^(?:©\s*Kancelaria Sejmu(?:\s+s\.\s*\d+/\d+)?"
    r"|s\.\s*\d+/\d+"
    r"|\d{4}-\d{2}-\d{2}"  # generation date of a unified text
    r"|(?:Dziennik Ustaw|Monitor Polski)\s*[–-]\s*\d+\s*[–-]\s*Poz\.\s*\d+)$",
    re.IGNORECASE,
)
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
# Unified texts mark amendments: "<…>" wording not yet in force, "[…]" wording in force
# today but due to change. See docs/adr/0002-current-act-text-source.md.
_FUTURE_WORDING = re.compile(r"<[^<>]*>")

# HTML elements outside the act's text: table of contents, footnote pop-ups, hidden citations.
_HTML_NOISE = "#toc, .show-all, .tooltip-text, .gloss-link, .xHidden, script, style"
_HTML_BLOCKS = ["div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table", "section"]


class ExtractedText(BaseModel):
    """Extracted plain text (one line per visual line) and extraction warnings."""

    text: str
    warnings: list[str] = Field(default_factory=list)


def extract_text(content: ActContent) -> ExtractedText:
    """Extract plain text from downloaded act content."""
    if content.media_type == "text/html":
        return extract_html_text(content.data)
    return extract_pdf_text(content.data)


def extract_pdf_text(data: bytes) -> ExtractedText:
    """Extract body text from a PDF page by page, keeping the top-to-bottom reading order."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [_text_spans(page) for page in doc]
    doc_size = _dominant_size([span for spans in pages for span in spans])
    page_lines: list[list[str]] = []
    dropped: list[dict] = []
    for spans in pages:
        body, small = _split_by_size(spans, doc_size)
        page_lines.append(_visual_lines(body))
        dropped += small
    lines = [line for line in _drop_repeated_edge_lines(page_lines) if not _FURNITURE.match(line)]
    text, warnings = _resolve_amendment_markers("\n".join(lines))

    warnings += _structure_warnings(text)
    total = sum(len(span["text"].strip()) for spans in pages for span in spans)
    dropped_chars = sum(len(span["text"].strip()) for span in dropped)
    if total and dropped_chars / total > _MAX_DROPPED_SHARE:
        warnings.append(
            f"Ekstrakcja PDF pominęła {dropped_chars / total:.0%} tekstu jako przypisy i elementy strony; "
            "układ dokumentu mógł się zmienić"
        )
    if suspicious := next((s["text"].strip() for s in dropped if _UNIT_START.match(s["text"].strip())), None):
        warnings.append(f"Ekstrakcja PDF pominęła tekst wyglądający na przepis: {suspicious[:80]!r}")
    return ExtractedText(text=text, warnings=warnings)


def _drop_repeated_edge_lines(page_lines: list[list[str]]) -> list[str]:
    """Drop running headers/footers: first/last page lines repeated (digits aside) on most pages."""

    def key(line: str) -> str:
        return _DIGITS.sub("#", line)

    def edges(lines: list[str]) -> list[str]:
        return lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]

    pages_with = Counter(k for lines in page_lines for k in {key(line) for line in edges(lines)})
    threshold = max(_MIN_REPEAT_PAGES, _MIN_REPEAT_SHARE * len(page_lines))
    repeated = {k for k, count in pages_with.items() if count >= threshold}

    kept: list[str] = []
    for lines in page_lines:
        last = len(lines) - _EDGE_LINES
        kept += [
            line
            for i, line in enumerate(lines)
            if not ((i < _EDGE_LINES or i >= last) and key(line) in repeated and not _UNIT_START.match(line))
        ]
    return kept


def _resolve_amendment_markers(text: str) -> tuple[str, list[str]]:
    """Keep the wording in force today: drop "<…>" blocks, unwrap "[…]".

    Markers that are unbalanced or nested are left untouched with a warning, so a stray
    "<" (e.g. a less-than sign) cannot swallow a large part of the act.
    """
    if not _balanced(text, "<", ">") or not _balanced(text, "[", "]"):
        return text, ["Niezrównoważone znaczniki zmian „<…>” / „[…]” w tekście aktu; pozostawiono je bez zmian"]
    text = _FUTURE_WORDING.sub("", text).replace("[", "").replace("]", "")
    lines = (_WHITESPACE.sub(" ", _INVISIBLE.sub("", line)).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line), []


def _balanced(text: str, opening: str, closing: str) -> bool:
    depth = 0
    for char in text:
        if char == opening:
            depth += 1
            if depth > 1:
                return False
        elif char == closing:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _structure_warnings(text: str) -> list[str]:
    if _UNIT_LINE.search(text):
        return []
    return ["W wyekstrahowanym tekście nie znaleziono jednostek „Art.” ani „§”"]


def _text_spans(page: pymupdf.Page) -> list[dict]:
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]


def _dominant_size(spans: list[dict]) -> float:
    sizes = Counter()
    for span in spans:
        sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _split_by_size(spans: list[dict], doc_size: float) -> tuple[list[dict], list[dict]]:
    """Split a page's spans into body text and smaller text (footnotes, notes, markers)."""
    has_body_text = any(span["size"] >= doc_size - _SIZE_TOLERANCE for span in spans)
    body_size = doc_size if has_body_text else _dominant_size(spans)
    body = [span for span in spans if span["size"] >= body_size - _SIZE_TOLERANCE]
    small = [span for span in spans if span["size"] < body_size - _SIZE_TOLERANCE]
    return body, small


def _visual_lines(body: list[dict]) -> list[str]:
    def centre(span: dict) -> float:
        return (span["bbox"][1] + span["bbox"][3]) / 2

    lines: list[list[dict]] = []
    for span in sorted(body, key=lambda s: (centre(s), s["bbox"][0])):
        if lines and abs(centre(span) - centre(lines[-1][0])) < _LINE_TOLERANCE:
            lines[-1].append(span)
        else:
            lines.append([span])
    return [_join_spans(sorted(line, key=lambda s: s["bbox"][0])) for line in lines]


def _join_spans(spans: list[dict]) -> str:
    text = spans[0]["text"]
    for previous, span in zip(spans, spans[1:]):
        gap = span["bbox"][0] - previous["bbox"][2]
        text += (" " if gap > _WORD_GAP else "") + span["text"]
    return _WHITESPACE.sub(" ", text).strip()


def extract_html_text(data: bytes) -> ExtractedText:
    """Extract text from the ELI act HTML, one structural unit (article, point, ...) per line."""
    soup = BeautifulSoup(data, "html.parser")
    root = soup.body or soup  # skip <head> and the doctype
    for element in root.select(_HTML_NOISE):
        element.decompose()
    # Source formatting wraps text nodes mid-sentence: whitespace inside a node is not a break.
    for node in root.find_all(string=True):
        node.replace_with(_WHITESPACE.sub(" ", node))
    for br in root.find_all("br"):
        br.replace_with("\n")
    for span in root.select("h1 > span"):  # act type, date and title are adjacent spans
        span.insert_after(NavigableString(" "))
    for element in root.find_all(_HTML_BLOCKS):
        # A unit's number ("Art. 27.", "1)") is an <h3> followed by its text: keep them on one line.
        is_unit_number = element.name == "h3" and "unit" in (element.parent.get("class") or [])
        element.insert_after(NavigableString(" " if is_unit_number else "\n"))
    lines = (_WHITESPACE.sub(" ", line).strip() for line in root.get_text().split("\n"))
    text = "\n".join(line for line in lines if line)
    return ExtractedText(text=text, warnings=_structure_warnings(text))
