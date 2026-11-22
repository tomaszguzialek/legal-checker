import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from legal_checker.config import Settings
from legal_checker.eli_client import ActCache, ActMetadata, EliClient, EliClientError, TextSource
from legal_checker.models import EliRef

# Mocked host: ".test" is reserved (RFC 2606), so these tests never reach the real ELI API.
BASE_URL = "https://eli-api.test/eli"
FIXTURES = Path(__file__).parent / "fixtures" / "eli"
CONSUMER_RIGHTS = EliRef(publisher="DU", year=2014, position=827)


@pytest.fixture
def client():
    with EliClient(BASE_URL) as eli:
        yield eli


@pytest.fixture
def metadata_json() -> dict:
    # Trimmed from a real GET /acts/DU/2014/827 response.
    return json.loads((FIXTURES / "DU_2014_827.json").read_text(encoding="utf-8"))


@respx.mock
def test_get_metadata(client, metadata_json):
    respx.get(f"{BASE_URL}/acts/DU/2014/827").respond(json=metadata_json)

    meta = client.get_metadata(CONSUMER_RIGHTS)

    assert meta.eli == "DU/2014/827"
    assert meta.title == "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta"
    assert meta.type == "Ustawa"
    assert meta.status == "akt posiada tekst jednolity"
    assert meta.in_force == "IN_FORCE"
    assert meta.entry_into_force == "2014-12-25"
    assert meta.text_html and meta.text_pdf
    assert [(t.type, t.file_name) for t in meta.texts][-1] == ("U", "D20140827Lj.pdf")
    assert meta.references["Inf. o tekście jednolitym"][0] == "DU/2026/1244"


@respx.mock
def test_get_metadata_tolerates_missing_optional_fields(client):
    respx.get(f"{BASE_URL}/acts/MP/2023/100").respond(
        json={"ELI": "MP/2023/100", "title": "Uchwała", "type": "Uchwała", "status": "obowiązujący"}
    )

    meta = client.get_metadata(EliRef(publisher="MP", year=2023, position=100))

    assert meta.texts == [] and meta.references == {} and meta.in_force is None


@respx.mock
@pytest.mark.parametrize(("status", "match"), [(404, "Nie znaleziono"), (500, "błąd 500")])
def test_get_metadata_http_errors(client, status, match):
    respx.get(f"{BASE_URL}/acts/DU/2014/827").respond(status)

    with pytest.raises(EliClientError, match=match):
        client.get_metadata(CONSUMER_RIGHTS)


@respx.mock
def test_get_metadata_network_error(client):
    respx.get(f"{BASE_URL}/acts/DU/2014/827").mock(side_effect=httpx.ConnectTimeout("timeout"))

    with pytest.raises(EliClientError, match="Błąd połączenia"):
        client.get_metadata(CONSUMER_RIGHTS)


def test_from_settings_uses_configured_base_url():
    with EliClient.from_settings(Settings(eli_api_base_url=BASE_URL)) as eli:
        assert str(eli.http.base_url) == f"{BASE_URL}/"


def _meta(texts: list[tuple[str, str]], html: bool = False, pdf: bool = True, consolidated: list[str] = ()) -> ActMetadata:
    return ActMetadata.from_api(
        {
            "ELI": "DU/2010/1",
            "title": "Ustawa testowa",
            "type": "Ustawa",
            "status": "obowiązujący",
            "textHTML": html,
            "textPDF": pdf,
            "texts": [{"type": t, "fileName": f} for t, f in texts],
            "references": {"Inf. o tekście jednolitym": [{"id": e} for e in consolidated]},
        }
    )


def test_select_text_source_prefers_unified_pdf_over_html(client, metadata_json):
    source = client.select_text_source(ActMetadata.from_api(metadata_json))

    assert (source.eli, source.kind, source.path) == (
        "DU/2014/827",
        "unified_pdf",
        "/acts/DU/2014/827/text/U/D20140827Lj.pdf",
    )


@respx.mock
@pytest.mark.parametrize(
    ("notice_texts", "expected_path"),
    [
        ([("O", "D20231.pdf"), ("T", "D20231L.pdf"), ("U", "D20231Lj.pdf")], "/acts/DU/2023/1/text/U/D20231Lj.pdf"),
        ([("O", "D20231.pdf"), ("I", "D20231.pdf")], "/acts/DU/2023/1/text/O/D20231.pdf"),
    ],
)
def test_select_text_source_falls_back_to_newest_consolidated_text(client, notice_texts, expected_path):
    notice_route = respx.get(f"{BASE_URL}/acts/DU/2023/1").respond(
        json={
            "ELI": "DU/2023/1",
            "title": "Obwieszczenie",
            "type": "Obwieszczenie",
            "status": "obowiązujący",
            "texts": [{"type": t, "fileName": f} for t, f in notice_texts],
        }
    )
    meta = _meta([("H", "text.html"), ("O", "D20101.pdf")], html=True, consolidated=["DU/2019/7", "DU/2023/1", "DU/2022/900"])

    source = client.select_text_source(meta)

    assert (source.eli, source.kind, source.path) == ("DU/2023/1", "consolidated_pdf", expected_path)
    assert notice_route.called


@pytest.mark.parametrize(
    ("html", "pdf", "kind", "path"),
    [(True, True, "html", "/acts/DU/2010/1/text.html"), (False, True, "pdf", "/acts/DU/2010/1/text.pdf")],
)
def test_select_text_source_without_amendments(client, html, pdf, kind, path):
    source = client.select_text_source(_meta([("O", "D20101.pdf")], html=html, pdf=pdf))

    assert (source.kind, source.path) == (kind, path)


def test_select_text_source_no_content(client):
    with pytest.raises(EliClientError, match="nie ma dostępnej treści"):
        client.select_text_source(_meta([], html=False, pdf=False))


@respx.mock
@pytest.mark.parametrize(
    ("kind", "path", "content_type", "media_type"),
    [
        ("unified_pdf", "/acts/DU/2014/827/text/U/D20140827Lj.pdf", "application/pdf", "application/pdf"),
        ("html", "/acts/DU/2010/1/text.html", "text/html;charset=utf-8", "text/html"),
    ],
)
def test_download_text(client, kind, path, content_type, media_type):
    body = b"%PDF-1.7 fake" if media_type == "application/pdf" else "<html>Art. 1.</html>".encode()
    respx.get(f"{BASE_URL}{path}").respond(content=body, headers={"content-type": content_type})
    source = TextSource(eli="DU/2014/827", kind=kind, path=path)

    content = client.download_text(source)

    assert content.data == body
    assert content.media_type == media_type
    assert content.url == f"{BASE_URL}{path}"
    assert content.source == source


@respx.mock
def test_download_text_rejects_unexpected_media_type(client):
    path = "/acts/DU/2014/827/text/U/D20140827Lj.pdf"
    respx.get(f"{BASE_URL}{path}").respond(text="<html>Błąd</html>", headers={"content-type": "text/html"})

    with pytest.raises(EliClientError, match="text/html zamiast application/pdf"):
        client.download_text(TextSource(eli="DU/2014/827", kind="unified_pdf", path=path))


@respx.mock
def test_download_text_not_found(client):
    path = "/acts/DU/2014/827/text/U/missing.pdf"
    respx.get(f"{BASE_URL}{path}").respond(404)

    with pytest.raises(EliClientError, match="Nie znaleziono"):
        client.download_text(TextSource(eli="DU/2014/827", kind="unified_pdf", path=path))


UNIFIED_PDF_PATH = "/acts/DU/2014/827/text/U/D20140827Lj.pdf"
PDF_BYTES = b"%PDF-1.7 fake unified text"


def _mock_consumer_rights_act(metadata_json: dict) -> tuple[respx.Route, respx.Route]:
    meta_route = respx.get(f"{BASE_URL}/acts/DU/2014/827").respond(json=metadata_json)
    pdf_route = respx.get(f"{BASE_URL}{UNIFIED_PDF_PATH}").respond(
        content=PDF_BYTES, headers={"content-type": "application/pdf"}
    )
    return meta_route, pdf_route


@respx.mock
def test_fetch_act_without_cache(client, metadata_json):
    meta_route, pdf_route = _mock_consumer_rights_act(metadata_json)

    act = client.fetch_act(CONSUMER_RIGHTS)
    client.fetch_act(CONSUMER_RIGHTS)

    assert act.metadata.eli == "DU/2014/827"
    assert act.content.data == PDF_BYTES
    assert act.content.url == f"{BASE_URL}{UNIFIED_PDF_PATH}"
    assert act.fetched_at.date() == datetime.now(UTC).date()
    assert meta_route.call_count == pdf_route.call_count == 2


@respx.mock
def test_fetch_act_cached_in_memory_and_on_disk(tmp_path, metadata_json):
    meta_route, pdf_route = _mock_consumer_rights_act(metadata_json)

    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:
        first = eli.fetch_act(CONSUMER_RIGHTS)
        assert eli.fetch_act(CONSUMER_RIGHTS) is first  # RAM hit

    entry = tmp_path / "eli" / "DU_2014_827"
    assert (entry / "content.pdf").read_bytes() == PDF_BYTES
    assert json.loads((entry / "act.json").read_text(encoding="utf-8"))["metadata"]["ELI"] == "DU/2014/827"

    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:  # fresh RAM, same directory
        from_disk = eli.fetch_act(CONSUMER_RIGHTS)

    assert from_disk == first
    assert meta_route.call_count == pdf_route.call_count == 1


@respx.mock
def test_fetch_act_refetches_expired_entry(tmp_path, metadata_json):
    meta_route, pdf_route = _mock_consumer_rights_act(metadata_json)
    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:
        stale = eli.fetch_act(CONSUMER_RIGHTS).model_copy(update={"fetched_at": datetime.now(UTC) - timedelta(days=2)})
        eli.cache.put(CONSUMER_RIGHTS, stale)

    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:
        assert eli.fetch_act(CONSUMER_RIGHTS).fetched_at > stale.fetched_at
    with EliClient(BASE_URL, cache=ActCache(tmp_path, max_age=None)) as eli:
        eli.cache.put(CONSUMER_RIGHTS, stale)
        assert eli.fetch_act(CONSUMER_RIGHTS) == stale  # never expires

    assert meta_route.call_count == pdf_route.call_count == 2


@respx.mock
@pytest.mark.parametrize("damage", ["corrupt_json", "missing_content"])
def test_fetch_act_ignores_damaged_cache_entry(tmp_path, metadata_json, damage):
    meta_route, _ = _mock_consumer_rights_act(metadata_json)
    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:
        eli.fetch_act(CONSUMER_RIGHTS)
    entry = tmp_path / "eli" / "DU_2014_827"
    if damage == "corrupt_json":
        (entry / "act.json").write_text("{not json", encoding="utf-8")
    else:
        (entry / "content.pdf").unlink()

    with EliClient(BASE_URL, cache=ActCache(tmp_path)) as eli:
        assert eli.fetch_act(CONSUMER_RIGHTS).content.data == PDF_BYTES

    assert meta_route.call_count == 2


def test_from_settings_enables_cache_in_cache_dir(tmp_path):
    with EliClient.from_settings(Settings(eli_api_base_url=BASE_URL, cache_dir=tmp_path)) as eli:
        assert eli.cache is not None and eli.cache.root == tmp_path / "eli"


@respx.mock
def test_derived_data_lives_and_dies_with_the_act_entry(tmp_path, metadata_json):
    meta_route, _ = _mock_consumer_rights_act(metadata_json)
    cache = ActCache(tmp_path)
    cache.put_derived(CONSUMER_RIGHTS, "extracted.json", "ignored")  # act not cached yet: no-op
    assert cache.get_derived(CONSUMER_RIGHTS, "extracted.json") is None

    with EliClient(BASE_URL, cache=cache) as eli:
        act = eli.fetch_act(CONSUMER_RIGHTS)
    cache.put_derived(CONSUMER_RIGHTS, "extracted.json", '{"text": "Art. 1."}')

    assert cache.get_derived(CONSUMER_RIGHTS, "extracted.json") == '{"text": "Art. 1."}'
    assert ActCache(tmp_path).get_derived(CONSUMER_RIGHTS, "extracted.json") == '{"text": "Art. 1."}'  # from disk
    assert ActCache(tmp_path).get_derived(CONSUMER_RIGHTS, "other.json") is None

    stale = act.model_copy(update={"fetched_at": datetime.now(UTC) - timedelta(days=2)})
    expired_cache = ActCache(tmp_path)
    expired_cache._memory[CONSUMER_RIGHTS.key] = stale
    assert expired_cache.get_derived(CONSUMER_RIGHTS, "extracted.json") is None  # act expired

    cache.put(CONSUMER_RIGHTS, act)  # act stored again (re-fetched): derived data is stale
    assert cache.get_derived(CONSUMER_RIGHTS, "extracted.json") is None
    assert not list((tmp_path / "eli" / "DU_2014_827").glob("derived-*"))
    assert meta_route.call_count == 1
