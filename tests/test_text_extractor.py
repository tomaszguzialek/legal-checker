from datetime import UTC, datetime

import pymupdf
import pytest

from legal_checker import text_extractor
from legal_checker.eli_client import ActCache, ActContent, ActMetadata, FetchedAct, TextSource
from legal_checker.models import EliRef
from legal_checker.text_extractor import (
    EXTRACTOR_VERSION,
    extract_act_text,
    extract_html_text,
    extract_pdf_text,
    extract_text,
)


def _unified_text_pdf() -> bytes:
    """Two pages laid out like a Kancelaria Sejmu unified text (sizes and positions from the real PDF)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((71, 55), "(c)Kancelaria Sejmu", fontsize=9)
    page.insert_text((444, 55), "s. 1/2", fontsize=9)
    page.insert_text((215, 230), "USTAWA", fontsize=12)
    # As in a real act, body text (12 pt) far outweighs notes and footnotes (10 pt); otherwise
    # the dropped-share sanity check fires (real PDFs drop 2-6% of their text).
    page.insert_text((96, 360), "Art. 1. Ustawa okresla prawa przyslugujace konsumentowi", fontsize=12)
    page.insert_text((415, 355), "1)", fontsize=7)  # superscript footnote marker
    page.insert_text((71, 380), "1)", fontsize=12)  # point number and text: separate spans, same line
    page.insert_text((96, 380), "obowiazki przedsiebiorcy zawierajacego umowe z konsumentem;", fontsize=12)
    page.insert_text((71, 400), "2) zasady i tryb zawierania z konsumentem umowy na", fontsize=12)
    page.insert_text((96, 420), "odleglosc i poza lokalem przedsiebiorstwa;", fontsize=12)
    page.insert_text((71, 440), "3) zasady i tryb wykonania prawa odstapienia od umowy zawartej", fontsize=12)
    page.insert_text((96, 460), "na odleglosc lub poza lokalem przedsiebiorstwa;", fontsize=12)
    page.insert_text((71, 480), "4) zasady i tryb zawierania z konsumentem umowy na odleglosc", fontsize=12)
    page.insert_text((96, 500), "dotyczacej uslug finansowych.", fontsize=12)
    page.insert_text((493, 400), "Dodany pkt 2 wejdzie w zycie", fontsize=10)  # margin note
    page.insert_text((71, 740), "1) Niniejsza ustawa wdraza dyrektywe 2011/83/UE.", fontsize=10)  # footnote
    page.insert_text((421, 795), "2026-10-29", fontsize=10)

    annex = doc.new_page(width=595, height=842)  # annexes are set entirely in 10 pt
    annex.insert_text((71, 55), "(c)Kancelaria Sejmu", fontsize=9)
    annex.insert_text((71, 100), "WZOR FORMULARZA ODSTAPIENIA OD UMOWY", fontsize=10)
    annex.insert_text((71, 120), "- Adresat", fontsize=10)
    annex.insert_text((90, 115), "(*)", fontsize=6.5)
    annex.insert_text((421, 795), "2026-10-29", fontsize=10)  # same size as the annex text
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_text_keeps_body_lines_in_reading_order():
    result = extract_pdf_text(_unified_text_pdf())

    assert result.warnings == []
    assert result.text.split("\n") == [
        "USTAWA",
        "Art. 1. Ustawa okresla prawa przyslugujace konsumentowi",
        "1) obowiazki przedsiebiorcy zawierajacego umowe z konsumentem;",
        "2) zasady i tryb zawierania z konsumentem umowy na",
        "odleglosc i poza lokalem przedsiebiorstwa;",
        "3) zasady i tryb wykonania prawa odstapienia od umowy zawartej",
        "na odleglosc lub poza lokalem przedsiebiorstwa;",
        "4) zasady i tryb zawierania z konsumentem umowy na odleglosc",
        "dotyczacej uslug finansowych.",
        "WZOR FORMULARZA ODSTAPIENIA OD UMOWY",
        "- Adresat",  # the annex-page footer "2026-10-29" (annex-sized) is removed by pattern
    ]


def test_extract_pdf_text_drops_page_furniture():
    text = extract_pdf_text(_unified_text_pdf()).text

    for noise in ("Kancelaria", "s. 1/2", "wejdzie w zycie", "Niniejsza ustawa", "(*)", "konsumentowi1)"):
        assert noise not in text


def test_extract_pdf_text_empty_document():
    doc = pymupdf.open()
    doc.new_page()
    assert extract_pdf_text(doc.tobytes()).text == ""


def _pdf(lines: list[tuple[str, float]]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, (text, size) in enumerate(lines):
        page.insert_text((71, 100 + 20 * i), text, fontsize=size)
    return doc.tobytes()


BODY = "Konsument moze odstapic od umowy w terminie 14 dni bez podania przyczyny."


def test_extract_pdf_text_drops_body_sized_running_headers_repeated_on_pages():
    doc = pymupdf.open()
    for number in range(1, 5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((200, 50), f"Dziennik Ustaw - {number + 1} - Poz. 827", fontsize=12)
        page.insert_text((71, 100), f"Art. {number}. (uchylony)", fontsize=12)
        page.insert_text((71, 120), BODY, fontsize=12)
        page.insert_text((71, 140), BODY, fontsize=12)
        page.insert_text((300, 800), f"Strona {number}", fontsize=12)

    lines = extract_pdf_text(doc.tobytes()).text.split("\n")

    assert lines[:3] == ["Art. 1. (uchylony)", BODY, BODY]
    assert len(lines) == 12
    assert not any("Dziennik Ustaw" in line or "Strona" in line for line in lines)


def test_extract_pdf_text_never_treats_repeated_provisions_as_headers():
    doc = pymupdf.open()
    for number in range(1, 5):  # every page starts with a repealed article: same line, digits aside
        page = doc.new_page(width=595, height=842)
        page.insert_text((71, 50), f"Art. {number}. (uchylony)", fontsize=12)
        page.insert_text((71, 70), BODY, fontsize=12)

    lines = extract_pdf_text(doc.tobytes()).text.split("\n")

    assert [line for line in lines if line.startswith("Art.")] == [f"Art. {n}. (uchylony)" for n in range(1, 5)]


@pytest.mark.parametrize(
    "furniture",
    # Built-in PDF fonts cannot draw an en dash; the pattern accepts both "–" and "-".
    ["Dziennik Ustaw - 2 - Poz. 827", "Monitor Polski - 3 - Poz. 100", "©Kancelaria Sejmu s. 1/48", "s. 2/48", "2026-10-29"],
)
def test_extract_pdf_text_drops_known_page_furniture_by_pattern(furniture):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((71, 50), furniture, fontsize=12, fontname="tiro")  # "tiro" can draw ©
    page.insert_text((71, 100), "Art. 1. " + BODY, fontsize=12)

    assert extract_pdf_text(doc.tobytes()).text == "Art. 1. " + BODY


def test_extract_pdf_text_keeps_wording_in_force_today():
    result = extract_pdf_text(
        _pdf(
            [
                ("Art. 2. Okreslenia uzyte w ustawie oznaczaja:", 12),
                ("5g) interoperacyjnosc - zdolnosc tresci cyfrowej;", 12),
                ("<5ga) aktualizacja oprogramowania - bezplatna aktualizacje,", 12),  # not yet in force
                ("niezbedna do zapewnienia zgodnosci z umowa;>", 12),
                ("[5) odpowiedzialnosc za zgodnosc swiadczenia z umowa;]", 12),  # in force, to be replaced
                ("<5) odpowiedzialnosc za zgodnosc swiadczenia z umowa, w tym okres jej trwania;>", 12),
                ("6) tresc uslug posprzedaznych [i gwarancji];", 12),
            ]
        )
    )

    assert result.warnings == []
    assert result.text.split("\n") == [
        "Art. 2. Okreslenia uzyte w ustawie oznaczaja:",
        "5g) interoperacyjnosc - zdolnosc tresci cyfrowej;",
        "5) odpowiedzialnosc za zgodnosc swiadczenia z umowa;",
        "6) tresc uslug posprzedaznych i gwarancji;",
    ]


@pytest.mark.parametrize(
    "line",
    ["masa pojazdu < 3,5 t i dlugosc pojazdu ponizej 12 m", "<a <b> c>", "[niedomkniety nawias", "a] b ["],
)
def test_extract_pdf_text_leaves_unbalanced_markers_untouched(line):
    result = extract_pdf_text(_pdf([("Art. 1. " + BODY, 12), (line, 12)]))

    assert result.text.split("\n")[1] == line
    assert any("Niezrównoważone znaczniki" in w for w in result.warnings)


def test_extract_pdf_text_joins_words_split_across_lines():
    result = extract_pdf_text(
        _pdf(
            [
                ("Art. 3. Obowiazki przedsie-", 12),
                ("biorcy wobec konsumenta okresla", 12),
                ("umowa. Adres e-", 12),  # real hyphen at a line break: repeated on the next line
                ("-mail podaje sie na stronie", 12),
                ("Art. 4. Termin - liczony od dnia", 12),  # spaced dash between words: not a split
                ("zawarcia umowy - wynosi 14 dni.", 12),
            ]
        )
    )

    assert result.text.split("\n") == [
        "Art. 3. Obowiazki przedsiebiorcy wobec konsumenta okresla",
        "umowa. Adres e-mail podaje sie na stronie",
        "Art. 4. Termin - liczony od dnia",
        "zawarcia umowy - wynosi 14 dni.",
    ]


def test_extract_pdf_text_keeps_split_before_capital_or_digit():
    lines = [("Art. 5. Zgodnie z ustawa-", 12), ("Kodeks cywilny oraz art. 12-", 12), ("14 ustawy.", 12)]

    assert extract_pdf_text(_pdf(lines)).text.split("\n") == [line for line, _ in lines]


def test_extract_pdf_text_warns_when_layout_breaks_the_size_filter():
    # Articles set smaller than the surrounding text: the filter drops them, which must not go unnoticed.
    result = extract_pdf_text(
        _pdf(
            [("Obwieszczenie Marszalka Sejmu w sprawie ogloszenia jednolitego tekstu ustawy", 12)] * 3
            + [("Art. 5. Konsument moze odstapic od umowy w terminie 14 dni.", 10)] * 2
        )
    )

    assert "Art. 5." not in result.text
    assert any("pominęła 3" in w and "układ dokumentu" in w for w in result.warnings)  # ~30% dropped
    assert any("wyglądający na przepis: 'Art. 5." in w for w in result.warnings)
    assert any("nie znaleziono jednostek" in w for w in result.warnings)


def test_extract_pdf_text_no_warning_for_a_normal_footnote_share():
    result = extract_pdf_text(_pdf([("Art. 1. Ustawa okresla prawa konsumenta i obowiazki.", 12)] * 9 + [("1) Przypis.", 10)]))

    assert result.warnings == []


ACT_HTML = """<!DOCTYPE HTML>
<html lang="pl"><head><meta charset="utf-8"><link rel="stylesheet" href="/act.css"></head>
<body>
  <h1><span class="head-type">Ustawa</span><span class="head-date">z dnia 30 maja 2014&nbsp;r.</span><span
     class="head-title">o prawach konsumenta <a class="gloss-link tooltip" href="#g"><sup>1)</sup><span
     class="tooltip-text">Niniejsza ustawa wdraża dyrektywę 2011/83/UE.</span></a></span></h1>
  <div id="toc"><h2>Spis treści</h2><ul class="toc"><li><a href="#chpt_1">Rozdział 1</a></li></ul></div>
  <div class="parts">
    <div class="unit unit_chpt" id="chpt_1"><h3><p align="center">&nbsp; Rozdział&nbsp;1 &nbsp;</p></h3>
      <div class="unit-inner">
        <div class="unit unit_arti" id="chpt_1-arti_27"><h3 class="pro-none"><b>Art.&nbsp;27.</b></h3>
          <div class="unit-inner"><div class="pro-text">Konsument, który zawarł umowę na odległość, może
                w terminie 14 dni odstąpić od niej, z wyjątkiem kosztów określonych w
                <a class="pro-lexlink" href="#a33">art. 33</a>.<span class="xHidden">(Dz. U. z 2014 r. poz. 827)</span>
          </div></div>
        </div>
        <div class="unit unit_pint" id="pint_1"><h3 class="pro-align-padding-right">1)</h3>
          <div class="unit-inner"><div class="pro-text">obowiązki przedsiębiorcy;</div></div>
        </div>
      </div>
    </div>
  </div>
  <div class="show-all hidden" id="show-all">Pokaż wszystko</div>
  <script>var x = 1;</script>
</body></html>"""


def test_extract_html_text():
    result = extract_html_text(ACT_HTML.encode())

    assert result.warnings == []
    assert result.text.split("\n") == [
        "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta",
        "Rozdział 1",
        "Art. 27. Konsument, który zawarł umowę na odległość, może w terminie 14 dni odstąpić od niej,"
        " z wyjątkiem kosztów określonych w art. 33.",
        "1) obowiązki przedsiębiorcy;",
    ]


def test_extract_html_text_warns_without_units():
    result = extract_html_text(b"<html><body><div>Regulamin spotkan zespolu.</div></body></html>")

    assert result.text == "Regulamin spotkan zespolu."
    assert result.warnings == ["W wyekstrahowanym tekście nie znaleziono jednostek „Art.” ani „§”"]


@pytest.mark.parametrize(
    ("kind", "media_type", "data", "expected_first_line"),
    [
        ("unified_pdf", "application/pdf", _unified_text_pdf(), "USTAWA"),
        ("html", "text/html", ACT_HTML.encode(), "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta"),
    ],
)
def test_extract_text_dispatches_on_media_type(kind, media_type, data, expected_first_line):
    content = ActContent(
        source=TextSource(eli="DU/2014/827", kind=kind, path="/acts/DU/2014/827/x"),
        url="https://eli-api.test/eli/acts/DU/2014/827/x",
        media_type=media_type,
        data=data,
    )

    assert extract_text(content).text.split("\n")[0] == expected_first_line


def test_extract_act_text_caches_result_next_to_the_act(tmp_path, monkeypatch):
    ref = EliRef(publisher="DU", year=2014, position=827)
    content = ActContent(
        source=TextSource(eli=ref.key, kind="html", path="/acts/DU/2014/827/text.html"),
        url="https://eli-api.test/eli/acts/DU/2014/827/text.html",
        media_type="text/html",
        data=ACT_HTML.encode(),
    )
    metadata = ActMetadata(ELI=ref.key, title="Ustawa", type="Ustawa", status="obowiązujący")
    cache = ActCache(tmp_path)
    cache.put(ref, FetchedAct(metadata=metadata, content=content, fetched_at=datetime.now(UTC)))
    calls = []
    monkeypatch.setattr(text_extractor, "extract_text", lambda c: calls.append(c) or extract_text(c))

    first = extract_act_text(ref, content, cache)
    from_ram = extract_act_text(ref, content, cache)
    from_disk = extract_act_text(ref, content, ActCache(tmp_path))
    without_cache = extract_act_text(ref, content)

    assert first == from_ram == from_disk == without_cache
    assert first.text.startswith("Ustawa z dnia 30 maja 2014 r.")
    assert len(calls) == 2  # first call and the uncached one
    assert (tmp_path / "eli" / "DU_2014_827" / f"derived-extracted-v{EXTRACTOR_VERSION}.json").exists()

    (tmp_path / "eli" / "DU_2014_827" / f"derived-extracted-v{EXTRACTOR_VERSION}.json").write_text("{broken")
    assert extract_act_text(ref, content, ActCache(tmp_path)) == first  # corrupted entry: extracted again
    assert len(calls) == 3
