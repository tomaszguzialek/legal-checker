"""Three-step resolver of an act hint to a canonical ELI key (SPEC Task 2.1)."""

import re

from legal_checker.models import EliRef

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
