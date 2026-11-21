"""Three-step resolver of an act hint to a canonical ELI key (SPEC Task 2.1)."""

import re

# Hyphen and dash look-alikes users paste from Word, PDFs and web pages.
_DASHES = re.compile(r"[‐‑‒–—―−﹘﹣－]")
_INVISIBLE = re.compile(r"[​‌‍⁠﻿]")
_WHITESPACE = re.compile(r"\s+")


def normalize_hint(hint: str) -> str:
    """Trim the hint, collapse whitespace and unify dashes to ``-``."""
    hint = _INVISIBLE.sub("", hint)
    hint = _DASHES.sub("-", hint)
    return _WHITESPACE.sub(" ", hint).strip()
