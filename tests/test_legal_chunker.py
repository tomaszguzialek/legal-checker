from legal_checker.legal_chunker import split_sections

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


def test_split_sections_keeps_a_stack_of_parent_headings():
    sections = split_sections(CIVIL_CODE)

    assert [(s.hierarchy_path, s.lines[0][:8]) for s in sections] == [
        ([], "Kodeks c"),
        (["Księga pierwsza CZĘŚĆ OGÓLNA", "Tytuł I Przepisy wstępne"], "Art. 1. "),
        (["Księga pierwsza CZĘŚĆ OGÓLNA", "Tytuł II Osoby", "Dział I Osoby fizyczne", "Rozdział I Zdolność prawna"], "Art. 8. "),
        (
            ["Księga pierwsza CZĘŚĆ OGÓLNA", "Tytuł II Osoby", "Dział I Osoby fizyczne", "Rozdział II Miejsce zamieszkania"],
            "Art. 25.",
        ),
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
