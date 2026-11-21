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
    lines: list[str] = []
    dropped: list[dict] = []
    for spans in pages:
        body, small = _split_by_size(spans, doc_size)
        lines += _visual_lines(body)
        dropped += small
    text = "\n".join(lines)

    warnings = _structure_warnings(text)
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
