import time
from pathlib import Path

import pytest

from legal_checker.bm25_retriever import InMemoryBm25Index, stem, tokenize
from legal_checker.legal_chunker import chunk_act
from legal_checker.models import LegalUnit


def test_tokenize_lowercases_and_drops_punctuation_and_stopwords():
    tokens = tokenize("Konsument, który zawarł umowę na odległość, może w terminie 14 dni od niej odstąpić (art. 27a).")

    assert tokens == ["konsument", "zawarł", "umow", "odległ", "moż", "termin", "14", "dni", "niej", "odstąp", "art", "27a"]


@pytest.mark.parametrize(
    "forms",
    [
        ["odstąpienie", "odstąpienia", "odstąpieniu", "odstąpić", "odstąpił"],
        ["konsument", "konsumenta", "konsumentowi", "konsumentem", "konsumentów"],
        ["umowa", "umowy", "umowie", "umową", "umowę", "umów"],
        ["przedsiębiorca", "przedsiębiorcy", "przedsiębiorcę", "przedsiębiorców"],
        ["odległość", "odległości"],
        ["sposób", "sposobu"],
    ],
)
def test_stem_maps_inflected_forms_to_one_stem(forms):
    assert len({stem(form) for form in forms}) == 1


def test_stem_keeps_short_words_and_numbers():
    assert [stem("dni"), stem("art"), stem("27a"), stem("449")] == ["dni", "art", "27a", "449"]


def unit(article: str, text: str, hierarchy_path: list[str] | None = None) -> LegalUnit:
    return LegalUnit(
        unit_id=f"DU/2014/827:art.{article}",
        eli="DU/2014/827",
        article=article,
        hierarchy_path=[*(hierarchy_path or []), f"Art. {article}"],
        text=text,
    )


def test_index_tokenizes_each_unit_with_its_parent_headings():
    units = [
        unit("1", "Ustawa określa prawa konsumenta."),
        unit("27", "Art. 27. Konsument może odstąpić od umowy.", ["Rozdział 4 Prawo odstąpienia"]),
    ]

    index = InMemoryBm25Index(units)

    assert index.units == units
    assert index.corpus[1] == [*tokenize("Rozdział 4 Prawo odstąpienia Art. 27"), "art", "27", "konsument", "moż", "odstąp", "umow"]
    assert index.bm25.corpus_size == 2


def test_index_rejects_an_act_without_units():
    with pytest.raises(ValueError):
        InMemoryBm25Index([])


def paragraph_unit(article: str, paragraph: str, text: str) -> LegalUnit:
    return unit(article, text).model_copy(
        update={
            "unit_id": f"DU/2014/827:art.{article}.ust.{paragraph}",
            "paragraph": paragraph,
            "hierarchy_path": [f"Art. {article}", f"ust. {paragraph}"],
        }
    )


ACT = [
    unit("1", "Art. 1. Ustawa określa prawa konsumenta oraz obowiązki przedsiębiorcy."),
    unit("12", "Art. 12. Przedsiębiorca informuje konsumenta o sposobie i terminie wykonania prawa odstąpienia od umowy."),
    unit("27", "Art. 27. Konsument, który zawarł umowę na odległość, może w terminie 14 dni odstąpić od niej."),
    unit("30", "Art. 30. Bieg terminu, o którym mowa w art. 27, rozpoczyna się od dnia zawarcia umowy."),
    unit("31", "Art. 31. Przedsiębiorca zwraca płatności, stosując sposób płatności użyty przez konsumenta."),
    unit("38", "Art. 38. Prawo odstąpienia nie przysługuje w umowach o świadczenie usług."),
    paragraph_unit("43", "1", "Art. 43. 1. Treść ustępu pierwszego."),
    paragraph_unit("43", "2", "Art. 43. 2. Treść ustępu drugiego."),
    unit("44", "Art. 44. Do umów stosuje się art. 385^1 Kodeksu cywilnego oraz art. 43."),
]


def test_search_ranks_units_sharing_the_query_terms_first():
    results = InMemoryBm25Index(ACT).search("termin na odstąpienie od umowy zawartej na odległość", top_k=3)

    articles = [u.article for u, _ in results]
    assert articles[0] == "27"
    assert [score for _, score in results] == sorted((score for _, score in results), reverse=True)


def test_search_leaves_out_units_without_any_query_term():
    results = InMemoryBm25Index(ACT).search("zwrot płatności")

    assert [u.article for u, _ in results] == ["31"]


def test_search_adds_articles_referenced_by_the_top_results():
    results = InMemoryBm25Index(ACT).search("bieg terminu", top_k=1)

    # Art. 30 refers to art. 27, which follows it although top_k is 1.
    assert [u.article for u, _ in results] == ["30", "27"]


def test_search_adds_every_paragraph_of_a_referenced_article_but_not_other_acts():
    results = InMemoryBm25Index(ACT).search("Kodeks cywilny", top_k=1)

    # Art. 44 refers to art. 43 (both units) and to art. 385^1 of another act (not in this one).
    assert [u.unit_id for u, _ in results] == ["DU/2014/827:art.44", "DU/2014/827:art.43.ust.1", "DU/2014/827:art.43.ust.2"]


def test_search_adds_only_the_referenced_paragraph_of_a_split_article():
    units = [*ACT, unit("45", "Art. 45. Wymogi, o których mowa w art. 43 ust. 2, dotyczą umów najmu.")]

    results = InMemoryBm25Index(units).search("umowa najmu", top_k=1)

    assert [u.unit_id for u, _ in results] == ["DU/2014/827:art.45", "DU/2014/827:art.43.ust.2"]


def test_search_does_not_duplicate_referenced_articles_already_found():
    results = InMemoryBm25Index(ACT).search("termin odstąpienia", top_k=12)

    articles = [u.article for u, _ in results]
    assert articles.count("27") == 1


def test_get_returns_a_unit_by_its_number():
    index = InMemoryBm25Index(ACT)

    assert index.get("27").unit_id == "DU/2014/827:art.27"
    assert index.get("Art. 27").unit_id == "DU/2014/827:art.27"
    assert index.get("99") is None


def test_get_returns_a_paragraph_or_the_whole_split_article():
    index = InMemoryBm25Index(ACT)

    assert index.get("43", "2").unit_id == "DU/2014/827:art.43.ust.2"
    assert index.get("43", "3") is None
    whole = index.get("43")
    assert whole.unit_id == "DU/2014/827:art.43"
    assert whole.paragraph is None
    assert whole.hierarchy_path == ["Art. 43"]
    assert whole.text == "Art. 43. 1. Treść ustępu pierwszego.\nArt. 43. 2. Treść ustępu drugiego."


def test_get_returns_an_article_kept_whole_for_any_of_its_paragraphs():
    assert InMemoryBm25Index(ACT).get("27", "1").unit_id == "DU/2014/827:art.27"


# Full text of the consumer rights act (DU/2014/827), extracted with the project's own pipeline
# from its unified-text PDF.
FULL_ACT = InMemoryBm25Index(
    chunk_act("DU/2014/827", (Path(__file__).parent / "fixtures" / "acts" / "DU_2014_827.txt").read_text()).units
)
# Sample clauses of an online shop's terms and conditions, each with the provision it must be checked against.
CLAUSES = [
    (
        "Klient może odstąpić od umowy zawartej przez Internet w terminie 7 dni od dnia otrzymania towaru, "
        "bez podania przyczyny.",
        "art.27",
    ),
    (
        "Sprzedawca zwróci Klientowi wpłacone środki w terminie 30 dni od dnia otrzymania oświadczenia "
        "o odstąpieniu od umowy.",
        "art.32",
    ),
    ("W przypadku odstąpienia od umowy koszty odesłania towaru do Sprzedawcy ponosi Klient.", "art.34"),
    (
        "Prawo odstąpienia od umowy nie przysługuje w odniesieniu do towarów wyprodukowanych według "
        "specyfikacji Klienta.",
        "art.38.ust.1",
    ),
    ("W razie niezgodności towaru z umową Klient może żądać naprawy lub wymiany towaru.", "art.43d"),
    ("Klient zrzeka się wszelkich praw przysługujących mu na podstawie ustawy.", "art.7"),
    (
        "Dostarczana treść cyfrowa podlega aktualizacjom, które Sprzedawca może wprowadzać według własnego uznania.",
        "art.43p",
    ),
]


@pytest.mark.parametrize(("clause", "provision"), CLAUSES, ids=[provision for _, provision in CLAUSES])
def test_search_finds_the_provision_for_a_contract_clause_in_the_top_5(clause, provision):
    top_5 = [u.unit_id for u, _ in FULL_ACT.search(clause)[:5]]

    assert f"DU/2014/827:{provision}" in top_5


def test_search_over_the_full_act_takes_under_50_ms():
    start = time.perf_counter()
    for clause, _ in CLAUSES:
        FULL_ACT.search(clause)
    elapsed_ms = (time.perf_counter() - start) * 1000 / len(CLAUSES)

    assert elapsed_ms < 50
