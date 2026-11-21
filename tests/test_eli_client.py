import json
from pathlib import Path

import httpx
import pytest
import respx

from legal_checker.config import Settings
from legal_checker.eli_client import EliClient, EliClientError
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
