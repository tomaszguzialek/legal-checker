import httpx
import pytest
import respx

from legal_checker.eli_resolver import (
    ActResolutionError,
    normalize_hint,
    parse_eli,
    parse_signature,
    resolve_hint,
    search_by_title,
)

BASE_URL = "https://api.sejm.gov.pl/eli"


def _act(eli: str, title: str, type_: str = "Ustawa", in_force: str = "IN_FORCE") -> dict:
    publisher, year, pos = eli.split("/")
    return {
        "ELI": eli,
        "publisher": publisher,
        "year": int(year),
        "pos": int(pos),
        "title": title,
        "type": type_,
        "inForce": in_force,
    }


# Trimmed from a real /acts/search?title=o prawach konsumenta response.
CONSUMER_RIGHTS_SEARCH = [
    _act("DU/2026/1353", "Ustawa z dnia 4 września 2026 r. o zmianie ustawy o przeciwdziałaniu "
         "nieuczciwym praktykom rynkowym oraz ustawy o prawach konsumenta"),
    _act("DU/2026/1244", "Obwieszczenie Marszałka Sejmu Rzeczypospolitej Polskiej z dnia 4 września 2026 r. "
         "w sprawie ogłoszenia jednolitego tekstu ustawy o prawach konsumenta", type_="Obwieszczenie"),
    _act("DU/2022/2581", "Ustawa z dnia 1 grudnia 2022 r. o zmianie ustawy o prawach konsumenta "
         "oraz niektórych innych ustaw"),
    _act("DU/2014/827", "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta"),
]

CIVIL_CODE_SEARCH = [
    _act("DU/1964/94", "Ustawa z dnia 23 kwietnia 1964 r. - Przepisy wprowadzające kodeks cywilny."),
    _act("DU/1964/93", "Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny."),
]

DATA_PROTECTION_SEARCH = [
    _act("DU/1997/883", "Ustawa z dnia 29 sierpnia 1997 r. o ochronie danych osobowych", in_force="NOT_IN_FORCE"),
    _act("DU/2018/1000", "Ustawa z dnia 10 maja 2018 r. o ochronie danych osobowych"),
]


@pytest.fixture
def http():
    with httpx.Client(base_url=BASE_URL) as client:
        yield client


def _mock_search(items: list[dict]) -> respx.Route:
    return respx.get(f"{BASE_URL}/acts/search").respond(json={"count": len(items), "items": items})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Dz.U. z 2014 r. poz. 827  ", "Dz.U. z 2014 r. poz. 827"),
        ("Dz.U.\tz 2014\nr.  poz. 827", "Dz.U. z 2014 r. poz. 827"),
        ("ustawa – Kodeks cywilny", "ustawa - Kodeks cywilny"),
        ("ustawa — Prawo pracy", "ustawa - Prawo pracy"),
        ("DU− 2014", "DU- 2014"),
        ("﻿RODO​", "RODO"),
    ],
)
def test_normalize_hint(raw, expected):
    assert normalize_hint(raw) == expected


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("DU/2014/827", "DU/2014/827"),
        ("du/2014/827", "DU/2014/827"),
        ("MP/2023/100", "MP/2023/100"),
        ("https://eli.gov.pl/eli/DU/2014/827/ogl", "DU/2014/827"),
        ("https://api.sejm.gov.pl/eli/acts/DU/2014/827/text.html", "DU/2014/827"),
        ("api.sejm.gov.pl/eli/acts/MP/2023/0100", "MP/2023/100"),
    ],
)
def test_parse_eli(hint, expected):
    assert parse_eli(hint).key == expected


@pytest.mark.parametrize(
    "hint",
    ["Dz.U. z 2014 r. poz. 827", "ustawa o prawach konsumenta", "XDU/2014/827", "DU/14/827", "DU/2014/"],
)
def test_parse_eli_no_match(hint):
    assert parse_eli(hint) is None


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("Dz.U. z 2014 r. poz. 827", "DU/2014/827"),
        ("Dz. U. 2014 poz. 827", "DU/2014/827"),
        ("Dz.U. 2014, nr 12, poz. 827", "DU/2014/827"),
        ("Dz. U. z 1964 r. Nr 16, poz. 93", "DU/1964/93"),
        ("M.P. 2023 poz. 100", "MP/2023/100"),
        ("M. P. z 2023 r. poz. 100", "MP/2023/100"),
        ("Dz.U.2014.827", "DU/2014/827"),
        ("dz.u. 2014 poz.827", "DU/2014/827"),
        ("Dziennik Ustaw z 2014 r. pozycja 827", "DU/2014/827"),
        ("Monitor Polski 2023 poz. 100", "MP/2023/100"),
        ("t.j. Dz. U. z 2023 r. poz. 2759 ze zm.", "DU/2023/2759"),
    ],
)
def test_parse_signature(hint, expected):
    assert parse_signature(hint).key == expected


@pytest.mark.parametrize(
    "hint",
    ["ustawa o prawach konsumenta", "Dz.U. z 2014 r.", "Dz.U. poz. 827", "DU/2014/827", "Kodeks pracy 1974 poz. 141"],
)
def test_parse_signature_no_match(hint):
    assert parse_signature(hint) is None



@respx.mock
@pytest.mark.parametrize(
    ("hint", "items", "expected"),
    [
        ("ustawa o prawach konsumenta", CONSUMER_RIGHTS_SEARCH, "DU/2014/827"),
        ("o prawach konsumenta", CONSUMER_RIGHTS_SEARCH, "DU/2014/827"),
        ("Kodeks cywilny", CIVIL_CODE_SEARCH, "DU/1964/93"),
        ("ustawa o ochronie danych osobowych", DATA_PROTECTION_SEARCH, "DU/2018/1000"),
    ],
)
def test_search_by_title_picks_best_candidate(http, hint, items, expected):
    route = _mock_search(items)

    assert search_by_title(hint, http).key == expected
    assert route.calls.last.request.url.params["title"] == hint


@respx.mock
def test_search_by_title_returns_amendment_when_asked(http):
    _mock_search(CONSUMER_RIGHTS_SEARCH)

    ref = search_by_title("o zmianie ustawy o prawach konsumenta oraz niektórych innych ustaw", http)

    assert ref.key == "DU/2022/2581"


@respx.mock
@pytest.mark.parametrize("items", [[], CONSUMER_RIGHTS_SEARCH])
def test_search_by_title_no_match(http, items):
    _mock_search(items)

    with pytest.raises(ActResolutionError, match="Nie znaleziono"):
        search_by_title("RODO", http)


@respx.mock
def test_search_by_title_http_error(http):
    respx.get(f"{BASE_URL}/acts/search").respond(503)

    with pytest.raises(ActResolutionError, match="nie powiodło się"):
        search_by_title("Kodeks cywilny", http)


@respx.mock
@pytest.mark.parametrize(
    "hint",
    ["https://eli.gov.pl/eli/DU/2014/827/ogl", "Dz.U. z 2014 r. poz. 827", "  ustawa  o prawach\u00a0konsumenta "],
)
def test_resolve_hint_all_paths(http, hint):
    route = _mock_search(CONSUMER_RIGHTS_SEARCH)

    assert resolve_hint(hint, http).key == "DU/2014/827"
    assert route.called == ("ustawa" in hint)


def test_resolve_hint_empty(http):
    with pytest.raises(ActResolutionError, match="Pusta"):
        resolve_hint(" \u200b ", http)


@respx.mock
def test_search_by_title_ignores_only_amendments_and_notices(http):
    _mock_search([item for item in CONSUMER_RIGHTS_SEARCH if item["ELI"] != "DU/2014/827"])

    with pytest.raises(ActResolutionError, match="Nie znaleziono"):
        search_by_title("ustawa o prawach konsumenta", http)


@respx.mock
def test_search_by_title_network_error(http):
    respx.get(f"{BASE_URL}/acts/search").mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(ActResolutionError, match="nie powiodło się"):
        search_by_title("Kodeks cywilny", http)


@respx.mock
@pytest.mark.parametrize("hint", ["Dz.U. z 2014 r.", "DU/14/827", "ustawa o czymś nieistniejącym"])
def test_resolve_hint_falls_through_to_title_search_and_fails(http, hint):
    route = _mock_search([])

    with pytest.raises(ActResolutionError, match="Nie znaleziono"):
        resolve_hint(hint, http)
    assert route.called


@respx.mock
def test_resolve_hint_normalizes_before_parsing(http):
    route = _mock_search([])

    assert resolve_hint("Dz.U. z 2014 r. poz. 827", http).key == "DU/2014/827"
    assert not route.called


@respx.mock
def test_resolve_hint_prefers_eli_key_over_signature(http):
    assert resolve_hint("DU/2014/827 (Dz.U. z 2023 r. poz. 2759)", http).key == "DU/2014/827"
