"""Three-step resolver of an act hint to a canonical ELI key (SPEC Task 2.1)."""

import re
from difflib import SequenceMatcher

import httpx

from legal_checker.models import EliRef


class ActResolutionError(ValueError):
    """The act hint could not be mapped to an ELI key."""


# Hyphen and dash look-alikes users paste from Word, PDFs and web pages.
_DASHES = re.compile(r"[‐‑‒–—―−﹘﹣－]")
_INVISIBLE = re.compile(r"[​‌‍⁠﻿]")
_WHITESPACE = re.compile(r"\s+")

# Bare key ("DU/2014/827") or the same path inside eli.gov.pl / api.sejm.gov.pl links.
_ELI_KEY = re.compile(r"(?<![A-Za-z])(DU|MP)/(\d{4})/(\d+)(?!\d)", re.IGNORECASE)

# Publication signature, e.g. "Dz.U. z 2014 r. poz. 827", "Dz. U. 2014, nr 12, poz. 827",
# "M.P. 2023 poz. 100" or the ISAP short form "Dz.U.2014.827". The issue number ("nr")
# of pre-2012 journals is skipped: ELI keys use the position only.
_SIGNATURE = re.compile(
    r"(?<![A-Za-z])(?P<publisher>Dz\.?\s?U|M\.?\s?P|Dziennik(?:a|u)?\s+Ustaw|Monitor(?:a|ze)?\s+Polski(?:m)?)\.?"
    r"[\s,]*(?:z\s+)?(?P<year>\d{4})(?:\s*r\.?)?"
    r"[\s,]*(?:nr\.?\s*\d+[\s,]*)?"
    r"(?:poz(?:\.|ycja)?\s*|\.)(?P<position>\d+)(?!\d)",
    re.IGNORECASE,
)

# Title search (path 3). The ELI API caps a page at 500 items and matches every query
# word as a substring, sorted by publisher rather than relevance, so ranking is ours.
_SEARCH_LIMIT = 500
_MIN_SIMILARITY = 0.6
_DATE_CLAUSE = re.compile(r"\s*z dnia \d{1,2} \S+ \d{4} r\.")
_ACT_TYPE_PREFIX = re.compile(r"^.*?z dnia \d{1,2} \S+ \d{4} r\.\s*(?:-\s*)?")
_AMENDMENT = re.compile(r"\bo zmianie\b|\bzmieniając", re.IGNORECASE)


def normalize_hint(hint: str) -> str:
    """Trim the hint, collapse whitespace and unify dashes to ``-``."""
    hint = _INVISIBLE.sub("", hint)
    hint = _DASHES.sub("-", hint)
    return _WHITESPACE.sub(" ", hint).strip()


def parse_eli(hint: str) -> EliRef | None:
    """Path 1: find an ELI key in a raw string or an ELI link."""
    match = _ELI_KEY.search(hint)
    if match is None:
        return None
    publisher, year, position = match.groups()
    return EliRef(publisher=publisher.upper(), year=int(year), position=int(position))


def parse_signature(hint: str) -> EliRef | None:
    """Path 2: map a Dz.U. / M.P. publication signature to an ELI key."""
    match = _SIGNATURE.search(hint)
    if match is None:
        return None
    publisher = "DU" if match["publisher"][0] in "Dd" else "MP"
    return EliRef(publisher=publisher, year=int(match["year"]), position=int(match["position"]))


def _similarity(query: str, title: str) -> float:
    """Best match of the query against the title without its date, or its subject alone."""
    query = query.lower().rstrip(".")
    title = title.lower().rstrip(".")
    dateless = _DATE_CLAUSE.sub("", title)
    subject = _ACT_TYPE_PREFIX.sub("", title)
    return max(SequenceMatcher(None, query, t).ratio() for t in (dateless, subject))


def search_by_title(hint: str, http: httpx.Client) -> EliRef:
    """Path 3: find the act via ``/acts/search?title=...`` and pick the best candidate.

    Consolidated-text announcements and amending acts are skipped (unless the hint asks
    for an amendment); ties are broken in favour of acts in force and statutes/regulations.
    """
    try:
        response = http.get("/acts/search", params={"title": hint, "limit": _SEARCH_LIMIT})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ActResolutionError(f"Wyszukiwanie aktu po tytule nie powiodło się: {exc}") from exc

    wants_amendment = _AMENDMENT.search(hint) is not None
    best: tuple[float, float, dict] | None = None
    for item in response.json().get("items", []):
        if item.get("type") == "Obwieszczenie":
            continue
        if not wants_amendment and _AMENDMENT.search(item["title"]):
            continue
        similarity = _similarity(hint, item["title"])
        score = (
            similarity
            + (0.1 if item.get("inForce") == "IN_FORCE" else 0.0)
            + (0.05 if item.get("type") in ("Ustawa", "Rozporządzenie") else 0.0)
        )
        if best is None or score > best[0]:
            best = (score, similarity, item)

    if best is None or best[1] < _MIN_SIMILARITY:
        raise ActResolutionError(f"Nie znaleziono aktu prawnego pasującego do wskazówki: {hint!r}")
    item = best[2]
    return EliRef(publisher=item["publisher"], year=item["year"], position=item["pos"])


def resolve_hint(hint: str, http: httpx.Client) -> EliRef:
    """Resolve an act hint to an ELI key: direct key/link, then signature, then title search."""
    hint = normalize_hint(hint)
    if not hint:
        raise ActResolutionError("Pusta wskazówka aktu prawnego")
    return parse_eli(hint) or parse_signature(hint) or search_by_title(hint, http)
