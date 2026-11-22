from legal_checker.legal_chunker import chunk_act, split_sections

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


def test_split_sections_caps_title_length():
    text = "Rozdział 3\nTytuł\nciąg dalszy\nkoniec tytułu\nTekst bez artykułu."

    (section,) = split_sections(text)

    assert section.hierarchy_path == ["Rozdział 3 Tytuł ciąg dalszy koniec tytułu"]
    assert section.lines == ["Tekst bez artykułu."]


def test_split_sections_accepts_superscript_designators():
    (section,) = split_sections("TYTUŁ VI^1\nOdpowiedzialność za produkt\nArt. 449^1. § 1. Treść.")

    assert section.hierarchy_path == ["Tytuł VI^1 Odpowiedzialność za produkt"]


def test_chunk_act_splits_articles_under_their_headings():
    units = chunk_act("DU/2024/1061", CIVIL_CODE)

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

    units = chunk_act("DU/2014/827", text)

    assert [u.unit_id for u in units] == ["DU/2014/827:art.449^1", "DU/2014/827:art.449^2"]
    assert units[0].text == "Art. 449^1. § 1. Kto wytwarza produkt.\n§ 2. Produktem jest rzecz."
    assert units[0].hierarchy_path == ["Art. 449^1"]


def test_chunk_act_handles_inserted_and_ranged_articles():
    text = "Art. 43k. Treść.\nArt. 44–50. (pominięte)\nArt. 51. Do umów zawartych\nprzed dniem wejścia w życie."

    units = chunk_act("DU/2014/827", text)

    assert [u.article for u in units] == ["43k", "44–50", "51"]
    assert units[1].text == "Art. 44–50. (pominięte)"
    assert units[2].text == "Art. 51. Do umów zawartych przed dniem wejścia w życie."


def test_chunk_act_ignores_article_references_at_line_start():
    text = "Art. 7. Przepis stosuje się do umów, o których mowa w\nart. 5, oraz\nArt. 6 ust. 2.\nArt. 8. Treść."

    assert [u.article for u in chunk_act("DU/2014/827", text)] == ["7", "8"]


def test_chunk_act_splits_regulations_by_paragraph_units():
    text = "ROZPORZĄDZENIE\nRozdział 1\nPrzepisy ogólne\n§ 1. Rozporządzenie określa wzór.\n§ 2. Wzór stanowi\nzałącznik."

    units = chunk_act("DU/2023/100", text)

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

    (unit,) = chunk_act("DU/2014/827", text)

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

    (unit,) = chunk_act("DU/2014/827", text)

    assert unit.text.split("\n") == [
        "Art. 2. Określenia użyte w ustawie oznaczają:",
        "1) umowa zawarta na odległość – umowę zawartą w dniu 14 maja 2014 r., o której mowa w art. 5 ust. 2 pkt 10–20.",
    ]


LONG = "Przedsiębiorca jest obowiązany poinformować konsumenta o prawie odstąpienia od umowy."


def test_chunk_act_splits_long_articles_into_paragraphs_with_article_context():
    text = f"Rozdział 4\nPrawo odstąpienia od umowy\nArt. 39. 1. {LONG}\n1) {LONG}\n2. {LONG}\n3. {LONG}\nArt. 40. 1. {LONG}"

    units = chunk_act("DU/2014/827", text, max_unit_chars=300)

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

    units = chunk_act("DU/2024/1061", text, max_unit_chars=200)

    assert [u.unit_id for u in units] == [f"DU/2024/1061:art.385^1.par.{n}" for n in (1, 2, 3)]
    assert units[2].text == f"Art. 385^1. § 3. {LONG}"
    assert units[2].hierarchy_path == ["Art. 385^1", "§ 3"]


def test_chunk_act_keeps_long_articles_without_paragraphs_whole():
    text = f"Art. 2. Określenia użyte w ustawie oznaczają:\n1) {LONG}\n2) {LONG}\n3) {LONG}"

    (unit,) = chunk_act("DU/2014/827", text, max_unit_chars=100)

    assert unit.unit_id == "DU/2014/827:art.2"
    assert unit.paragraph is None


def test_chunk_act_splits_long_regulation_units_by_subparagraphs():
    text = f"§ 3. 1. {LONG}\n2. {LONG}"

    units = chunk_act("DU/2023/100", text, max_unit_chars=100)

    assert [(u.unit_id, u.text[:8]) for u in units] == [
        ("DU/2023/100:par.3.ust.1", "§ 3. 1. "),
        ("DU/2023/100:par.3.ust.2", "§ 3. 2. "),
    ]
