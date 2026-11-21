import pytest

from legal_checker.eli_resolver import normalize_hint, parse_eli, parse_signature


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
