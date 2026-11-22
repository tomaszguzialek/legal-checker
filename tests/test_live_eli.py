"""Smoke test against the real Sejm ELI API (SPEC section 5, tip 2). Run with: uv run pytest -m live

It detects changes in the API's data or in the PDF layout that the mocked tests cannot see,
e.g. a layout change that breaks the font-size filter of the text extractor.
"""

import re

import pytest

from legal_checker.config import Settings
from legal_checker.eli_client import ActCache, EliClient
from legal_checker.eli_resolver import resolve_hint
from legal_checker.text_extractor import extract_act_text

pytestmark = pytest.mark.live


def test_consumer_rights_act_from_title_to_clean_text(tmp_path):
    with EliClient(Settings().eli_api_base_url, cache=ActCache(tmp_path)) as eli:
        ref = resolve_hint("ustawa o prawach konsumenta", eli.http)  # title search, path 3
        act = eli.fetch_act(ref)
        result = extract_act_text(ref, act.content, eli.cache)

    assert ref.key == "DU/2014/827"
    assert act.content.source.kind == "unified_pdf"
    assert result.warnings == []

    text = re.sub(r"\s+", " ", result.text)
    assert "Art. 1. Ustawa określa prawa przysługujące konsumentowi" in text
    assert "w terminie 14 dni odstąpić od niej bez podawania przyczyny" in text  # art. 27
    assert "WZÓR FORMULARZA ODSTĄPIENIA OD UMOWY" in text  # annex set in a smaller font
    # Page furniture and amendment markers are gone. (Not asserting on specific not-yet-in-force
    # wording: it legitimately appears once it enters into force.)
    for leftover in ("Kancelaria Sejmu", "<", ">", "[", "]"):
        assert leftover not in text
