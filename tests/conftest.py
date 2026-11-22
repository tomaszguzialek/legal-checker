from datetime import UTC, datetime
from pathlib import Path

import pytest

from legal_checker.eli_client import ActCache, ActContent, ActMetadata, EliClient, FetchedAct, TextSource
from legal_checker.models import EliRef
from legal_checker.text_extractor import EXTRACTOR_VERSION, ExtractedText

# Mocked host: ".test" is reserved (RFC 2606), so tests never reach the real ELI API.
ELI_TEST_URL = "https://eli-api.test/eli"
CONSUMER_RIGHTS = EliRef(publisher="DU", year=2014, position=827)
CONSUMER_RIGHTS_TEXT = (Path(__file__).parent / "fixtures" / "acts" / "DU_2014_827.txt").read_text(encoding="utf-8")


@pytest.fixture
def cached_act(tmp_path) -> FetchedAct:
    """The consumer rights act, fetched just now, in an ``ActCache`` at ``tmp_path``.

    Its extracted text (the fixture text, with one extraction warning) is cached next to it,
    so the graph reads the act without HTTP and without parsing a PDF.
    """
    path = "/acts/DU/2014/827/text/U/D20140827Lj.pdf"
    act = FetchedAct(
        metadata=ActMetadata(
            ELI="DU/2014/827",
            title="Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
            type="Ustawa",
            status="obowiązujący",
        ),
        content=ActContent(
            source=TextSource(eli="DU/2014/827", kind="unified_pdf", path=path),
            url=f"{ELI_TEST_URL}{path}",
            media_type="application/pdf",
            data=b"%PDF-1.7 fake",
        ),
        fetched_at=datetime.now(UTC),
    )
    cache = ActCache(tmp_path)
    cache.put(CONSUMER_RIGHTS, act)
    extracted = ExtractedText(text=CONSUMER_RIGHTS_TEXT, warnings=["Ostrzeżenie ekstrakcji"])
    cache.put_derived(CONSUMER_RIGHTS, f"extracted-v{EXTRACTOR_VERSION}.json", extracted.model_dump_json())
    return act


@pytest.fixture
def eli(tmp_path, cached_act):
    """ELI client on the mocked host, serving the consumer rights act from the cache."""
    with EliClient(ELI_TEST_URL, cache=ActCache(tmp_path)) as client:
        yield client
