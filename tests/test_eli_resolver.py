import pytest

from legal_checker.eli_resolver import normalize_hint


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
