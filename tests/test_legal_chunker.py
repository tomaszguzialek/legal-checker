import re
from pathlib import Path

from legal_checker.legal_chunker import FALLBACK_CHUNK_CHARS, NO_UNITS_WARNING, chunk_act, split_sections

# Heading layout of the Civil Code (DU/2024/1061) as extracted from its unified-text PDF.
CIVIL_CODE = """\
Kodeks cywilny
KSIĘGA PIERWSZA
CZĘŚĆ OGÓLNA
TYTUŁ I
Przepisy wstępne
Art. 1. Kodeks niniejszy reguluje stosunki cywilnoprawne.
TYTUŁ II
Osoby
DZIAŁ I
Osoby fizyczne
Rozdział I
Zdolność prawna
Art. 8. § 1. Każdy człowiek od chwili urodzenia ma zdolność prawną.
Rozdział II
Miejsce zamieszkania
Art. 25. Miejscem zamieszkania osoby fizycznej jest miejscowość,
w której ta osoba przebywa z zamiarem stałego pobytu.
KSIĘGA CZWARTA
SPADKI
TYTUŁ III
Testamenty
Oddział 1
Testamenty zwykłe
Art. 949. § 1. Spadkodawca może sporządzić testament pismem ręcznym."""


PERSONS = ["Księga pierwsza CZĘŚĆ OGÓLNA", "Tytuł II Osoby", "Dział I Osoby fizyczne"]


def test_split_sections_keeps_a_stack_of_parent_headings():
    sections = split_sections(CIVIL_CODE)

    assert [(s.hierarchy_path, s.lines[0][:8]) for s in sections] == [
        ([], "Kodeks c"),
        (["Księga pierwsza CZĘŚĆ OGÓLNA", "Tytuł I Przepisy wstępne"], "Art. 1. "),
        ([*PERSONS, "Rozdział I Zdolność prawna"], "Art. 8. "),
        ([*PERSONS, "Rozdział II Miejsce zamieszkania"], "Art. 25."),
        # A higher heading closes the lower ones; skipped levels (no DZIAŁ, no Rozdział) are fine.
        (["Księga czwarta SPADKI", "Tytuł III Testamenty", "Oddział 1 Testamenty zwykłe"], "Art. 949"),
    ]
    assert sections[3].lines == [
        "Art. 25. Miejscem zamieszkania osoby fizycznej jest miejscowość,",
        "w której ta osoba przebywa z zamiarem stałego pobytu.",
    ]


def test_split_sections_joins_multi_line_titles():
    text = "Rozdział 2\nUmowy zawierane poza lokalem\nprzedsiębiorstwa\nArt. 7. Treść."

    (section,) = split_sections(text)

    assert section.hierarchy_path == ["Rozdział 2 Umowy zawierane poza lokalem przedsiębiorstwa"]
    assert section.lines == ["Art. 7. Treść."]


def test_split_sections_handles_untitled_headings_and_regulation_paragraphs():
    sections = split_sections("Rozdział 1\n§ 1. Rozporządzenie określa wzór.\nRozdział 5a\nArt. 38a. Treść.")

    assert [(s.hierarchy_path, s.lines) for s in sections] == [
        (["Rozdział 1"], ["§ 1. Rozporządzenie określa wzór."]),
        (["Rozdział 5a"], ["Art. 38a. Treść."]),
    ]


def test_split_sections_ignores_heading_words_in_body_text():
    text = "Art. 43h. 1. Przepisy stosuje się, o ile\nrozdział 4\nnie stanowi inaczej, oraz Rozdziału 5."

    (section,) = split_sections(text)

    assert section.hierarchy_path == []
    assert len(section.lines) == 3


def test_split_sections_without_titles_keeps_lines_after_headings_as_body():
    sections = split_sections("Rozdział 1\nZasady ogólne\nTreść.", titles=False)

    assert [(s.hierarchy_path, s.lines) for s in sections] == [(["Rozdział 1"], ["Zasady ogólne", "Treść."])]


def test_split_sections_caps_title_length():
    text = "Rozdział 3\nTytuł\nciąg dalszy\nkoniec tytułu\nTekst bez artykułu."

    (section,) = split_sections(text)

    assert section.hierarchy_path == ["Rozdział 3 Tytuł ciąg dalszy koniec tytułu"]
    assert section.lines == ["Tekst bez artykułu."]


def test_split_sections_accepts_superscript_designators():
    (section,) = split_sections("TYTUŁ VI^1\nOdpowiedzialność za produkt\nArt. 449^1. § 1. Treść.")

    assert section.hierarchy_path == ["Tytuł VI^1 Odpowiedzialność za produkt"]


def test_chunk_act_splits_articles_under_their_headings():
    units = chunk_act("DU/2024/1061", CIVIL_CODE).units

    assert [(u.unit_id, u.article, u.hierarchy_path[-2:]) for u in units] == [
        ("DU/2024/1061:art.1", "1", ["Tytuł I Przepisy wstępne", "Art. 1"]),
        ("DU/2024/1061:art.8", "8", ["Rozdział I Zdolność prawna", "Art. 8"]),
        ("DU/2024/1061:art.25", "25", ["Rozdział II Miejsce zamieszkania", "Art. 25"]),
        ("DU/2024/1061:art.949", "949", ["Oddział 1 Testamenty zwykłe", "Art. 949"]),
    ]
    assert units[2].text == (
        "Art. 25. Miejscem zamieszkania osoby fizycznej jest miejscowość, "
        "w której ta osoba przebywa z zamiarem stałego pobytu."
    )
    assert all(u.eli == "DU/2024/1061" and u.paragraph is None for u in units)


def test_chunk_act_keeps_paragraphs_inside_articles_and_drops_the_act_title():
    text = "Dz. U. 2014 poz. 827\nUSTAWA\nArt. 449^1. § 1. Kto wytwarza produkt.\n§ 2. Produktem jest rzecz.\nArt. 449^2. Treść."

    units = chunk_act("DU/2014/827", text).units

    assert [u.unit_id for u in units] == ["DU/2014/827:art.449^1", "DU/2014/827:art.449^2"]
    assert units[0].text == "Art. 449^1. § 1. Kto wytwarza produkt.\n§ 2. Produktem jest rzecz."
    assert units[0].hierarchy_path == ["Art. 449^1"]


def test_chunk_act_handles_inserted_and_ranged_articles():
    text = "Art. 43k. Treść.\nArt. 44–50. (pominięte)\nArt. 51. Do umów zawartych\nprzed dniem wejścia w życie."

    units = chunk_act("DU/2014/827", text).units

    assert [u.article for u in units] == ["43k", "44–50", "51"]
    assert units[1].text == "Art. 44–50. (pominięte)"
    assert units[2].text == "Art. 51. Do umów zawartych przed dniem wejścia w życie."


def test_chunk_act_ignores_article_references_at_line_start():
    text = "Art. 7. Przepis stosuje się do umów, o których mowa w\nart. 5, oraz\nArt. 6 ust. 2.\nArt. 8. Treść."

    assert [u.article for u in chunk_act("DU/2014/827", text).units] == ["7", "8"]


def test_chunk_act_splits_regulations_by_paragraph_units():
    text = "ROZPORZĄDZENIE\nRozdział 1\nPrzepisy ogólne\n§ 1. Rozporządzenie określa wzór.\n§ 2. Wzór stanowi\nzałącznik."

    units = chunk_act("DU/2023/100", text).units

    assert [(u.unit_id, u.article, u.hierarchy_path) for u in units] == [
        ("DU/2023/100:par.1", "1", ["Rozdział 1 Przepisy ogólne", "§ 1"]),
        ("DU/2023/100:par.2", "2", ["Rozdział 1 Przepisy ogólne", "§ 2"]),
    ]
    assert units[1].text == "§ 2. Wzór stanowi załącznik."


def test_chunk_act_puts_each_paragraph_point_and_letter_on_its_own_line():
    text = """\
Art. 12. 1. Najpóźniej w chwili wyrażenia przez konsumenta woli związania się
umową przedsiębiorca ma obowiązek poinformować konsumenta o:
1) głównych cechach świadczenia z uwzględnieniem przedmiotu świadczenia oraz
sposobu porozumiewania się z konsumentem;
2) danych identyfikujących przedsiębiorcę, w szczególności:
a) firmie,
b) adresie;
2. Informacje, o których mowa w ust. 1 pkt 2, stanowią
integralną część umowy."""

    (unit,) = chunk_act("DU/2014/827", text).units

    assert unit.text.split("\n") == [
        "Art. 12. 1. Najpóźniej w chwili wyrażenia przez konsumenta woli związania się "
        "umową przedsiębiorca ma obowiązek poinformować konsumenta o:",
        "1) głównych cechach świadczenia z uwzględnieniem przedmiotu świadczenia oraz "
        "sposobu porozumiewania się z konsumentem;",
        "2) danych identyfikujących przedsiębiorcę, w szczególności:",
        "a) firmie,",
        "b) adresie;",
        "2. Informacje, o których mowa w ust. 1 pkt 2, stanowią integralną część umowy.",
    ]


def test_chunk_act_keeps_wrapped_lines_starting_with_numbers_or_dashes_in_place():
    text = (
        "Art. 2. Określenia użyte w ustawie oznaczają:\n1) umowa zawarta na odległość\n– umowę zawartą w dniu\n"
        "14 maja 2014 r., o której mowa w art.\n5 ust. 2 pkt 10–\n20."
    )

    (unit,) = chunk_act("DU/2014/827", text).units

    assert unit.text.split("\n") == [
        "Art. 2. Określenia użyte w ustawie oznaczają:",
        "1) umowa zawarta na odległość – umowę zawartą w dniu 14 maja 2014 r., o której mowa w art. 5 ust. 2 pkt 10–20.",
    ]


LONG = "Przedsiębiorca jest obowiązany poinformować konsumenta o prawie odstąpienia od umowy."


def test_chunk_act_splits_long_articles_into_paragraphs_with_article_context():
    text = f"Rozdział 4\nPrawo odstąpienia od umowy\nArt. 39. 1. {LONG}\n1) {LONG}\n2. {LONG}\n3. {LONG}\nArt. 40. 1. {LONG}"

    units = chunk_act("DU/2014/827", text, max_unit_chars=300).units

    assert [(u.unit_id, u.article, u.paragraph, u.hierarchy_path) for u in units] == [
        ("DU/2014/827:art.39.ust.1", "39", "1", ["Rozdział 4 Prawo odstąpienia od umowy", "Art. 39", "ust. 1"]),
        ("DU/2014/827:art.39.ust.2", "39", "2", ["Rozdział 4 Prawo odstąpienia od umowy", "Art. 39", "ust. 2"]),
        ("DU/2014/827:art.39.ust.3", "39", "3", ["Rozdział 4 Prawo odstąpienia od umowy", "Art. 39", "ust. 3"]),
        ("DU/2014/827:art.40", "40", None, ["Rozdział 4 Prawo odstąpienia od umowy", "Art. 40"]),
    ]
    assert units[0].text == f"Art. 39. 1. {LONG}\n1) {LONG}"
    assert units[1].text == f"Art. 39. 2. {LONG}"


def test_chunk_act_splits_long_code_articles_by_section_sign():
    text = f"Art. 385^1. § 1. {LONG}\n§ 2. {LONG}\n§ 3. {LONG}"

    units = chunk_act("DU/2024/1061", text, max_unit_chars=200).units

    assert [u.unit_id for u in units] == [f"DU/2024/1061:art.385^1.par.{n}" for n in (1, 2, 3)]
    assert units[2].text == f"Art. 385^1. § 3. {LONG}"
    assert units[2].hierarchy_path == ["Art. 385^1", "§ 3"]


def test_chunk_act_keeps_long_articles_without_paragraphs_whole():
    text = f"Art. 2. Określenia użyte w ustawie oznaczają:\n1) {LONG}\n2) {LONG}\n3) {LONG}"

    (unit,) = chunk_act("DU/2014/827", text, max_unit_chars=100).units

    assert unit.unit_id == "DU/2014/827:art.2"
    assert unit.paragraph is None


def test_chunk_act_splits_long_regulation_units_by_subparagraphs():
    text = f"§ 3. 1. {LONG}\n2. {LONG}"

    units = chunk_act("DU/2023/100", text, max_unit_chars=100).units

    assert [(u.unit_id, u.text[:8]) for u in units] == [
        ("DU/2023/100:par.3.ust.1", "§ 3. 1. "),
        ("DU/2023/100:par.3.ust.2", "§ 3. 2. "),
    ]


def test_chunk_act_warns_only_without_recognisable_units():
    assert chunk_act("DU/2024/1061", CIVIL_CODE).warnings == []
    assert chunk_act("DU/2023/100", "§ 1. Rozporządzenie określa wzór.").warnings == []

    result = chunk_act("MP/2023/100", "OBWIESZCZENIE\nOgłasza się tekst komunikatu.")

    assert result.warnings == [NO_UNITS_WARNING]
    assert [(u.unit_id, u.article, u.hierarchy_path, u.text) for u in result.units] == [
        ("MP/2023/100:akapit.1", "1", ["Akapit 1"], "OBWIESZCZENIE Ogłasza się tekst komunikatu."),
    ]


def test_chunk_act_fallback_packs_whole_paragraphs_into_chunks():
    # Each line ends a sentence and the next starts with a capital: a possible paragraph break.
    sentences = [f"Zdanie {n}. {LONG}" for n in range(30)]
    wrapped = ["Ostatni akapit zawinięty w środku", "zdania, który nie może zostać rozdzielony."]

    units = chunk_act("MP/2023/100", "\n".join(sentences + wrapped)).units

    assert len(units) > 1
    assert all(len(u.text) >= FALLBACK_CHUNK_CHARS for u in units[:-1])
    assert " ".join(u.text for u in units) == " ".join(sentences + wrapped)
    assert units[-1].text.endswith("w środku zdania, który nie może zostać rozdzielony.")
    assert [u.unit_id for u in units] == [f"MP/2023/100:akapit.{n}" for n in range(1, len(units) + 1)]


def test_chunk_act_fallback_keeps_parent_headings():
    text = "Wstęp komunikatu.\nRozdział 1\nZasady ogólne\nTreść zasad.\nRozdział 2\nPrzepisy końcowe\nTreść\nprzepisów."

    units = chunk_act("MP/2023/100", text).units

    assert [(u.hierarchy_path, u.text) for u in units] == [
        (["Akapit 1"], "Wstęp komunikatu."),
        # Without units nothing marks where a title ends: it stays in the text, never lost.
        (["Rozdział 1", "Akapit 2"], "Zasady ogólne Treść zasad."),
        (["Rozdział 2", "Akapit 3"], "Przepisy końcowe Treść przepisów."),
    ]


def test_chunk_act_empty_text():
    result = chunk_act("MP/2023/100", "")

    assert result.units == []
    assert result.warnings == [NO_UNITS_WARNING]


# Real multi-chapter fragment of the consumer rights act (DU/2014/827) as extracted from its
# unified-text PDF: chapters 1, 4, 6 and 7 with a long, a repealed and a range article.
FRAGMENT = (Path(__file__).parent / "fixtures" / "acts" / "DU_2014_827_fragment.txt").read_text()
CHAPTER_4 = "Rozdział 4 Prawo odstąpienia od umowy zawartej na odległość lub poza lokalem przedsiębiorstwa"


def test_chunk_act_on_a_multi_chapter_act_fragment():
    result = chunk_act("DU/2014/827", FRAGMENT)

    assert result.warnings == []
    assert [(u.unit_id, u.article, u.paragraph, u.hierarchy_path) for u in result.units] == [
        ("DU/2014/827:art.1", "1", None, ["Rozdział 1 Przepisy ogólne", "Art. 1"]),
        ("DU/2014/827:art.27", "27", None, [CHAPTER_4, "Art. 27"]),
        ("DU/2014/827:art.28", "28", None, [CHAPTER_4, "Art. 28"]),
        ("DU/2014/827:art.38.ust.1", "38", "1", [CHAPTER_4, "Art. 38", "ust. 1"]),  # over MAX_UNIT_CHARS
        ("DU/2014/827:art.38.ust.2", "38", "2", [CHAPTER_4, "Art. 38", "ust. 2"]),
        ("DU/2014/827:art.38a", "38a", None, [CHAPTER_4, "Art. 38a"]),
        ("DU/2014/827:art.44–50", "44–50", None, ["Rozdział 6 Przepisy zmieniające", "Art. 44–50"]),
        ("DU/2014/827:art.51", "51", None, ["Rozdział 7 Przepisy przejściowe i końcowe", "Art. 51"]),
    ]
    assert all(u.eli == "DU/2014/827" for u in result.units)


def test_chunk_act_keeps_full_provision_text_of_the_fragment():
    units = {u.unit_id.split(":")[1]: u.text for u in chunk_act("DU/2014/827", FRAGMENT).units}

    assert units["art.27"] == (
        "Art. 27. 1. Konsument, który zawarł umowę na odległość lub poza lokalem przedsiębiorstwa, może w terminie "
        "14 dni odstąpić od niej bez podawania przyczyny i bez ponoszenia kosztów, z wyjątkiem kosztów określonych "
        "w art. 33, art. 34 ust. 2 i art. 35.\n"
        "2. Jeżeli konsument zawarł umowę poza lokalem przedsiębiorstwa podczas nieumówionej wizyty przedsiębiorcy "
        "w miejscu zamieszkania lub zwykłego pobytu konsumenta albo wycieczki, termin do odstąpienia od umowy "
        "wynosi 30 dni."
    )
    assert [line[:3] for line in units["art.28"].split("\n")] == ["Art", "1) ", "a) ", "b) ", "2) "]
    assert [line[:3] for line in units["art.1"].split("\n")] == ["Art", "1) ", "2) ", "3) ", "4) ", "4a)", "4b)"]
    assert units["art.38.ust.2"].startswith("Art. 38. 2. Przepisów ust. 1 pkt 1–3 i 5 nie stosuje się")

    # Nothing but the act's title and the chapter headings is left out of the units.
    def words(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    headings = [
        "Dz. U. 2014 poz. 827 U S TAWA z dnia 30 maja 2014 r. o prawach konsumenta",
        "Rozdział 1 Przepisy ogólne",
        CHAPTER_4,
        "Rozdział 6 Przepisy zmieniające",
        "Rozdział 7 Przepisy przejściowe i końcowe",
    ]
    body = words(FRAGMENT)
    for heading in headings:
        assert heading in body
        body = body.replace(heading + " ", "", 1)
    assert words(" ".join(units.values()).replace("Art. 38. 2.", "2.")) == body
